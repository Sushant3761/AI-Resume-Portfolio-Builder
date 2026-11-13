import logging
import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import Mock, call, patch
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit.testing.v1 import AppTest

import app as streamlit_app
from utils import docx_generator, pdf_generator, resume_generator, similarity
from utils.portfolio_generator import build_portfolio_html


def make_response(status, body=None, headers=None):
    response = Mock()
    response.status_code = status
    response.headers = headers or {}
    response.json.return_value = body if body is not None else {}
    return response


class OpenRouterReliabilityTests(unittest.TestCase):
    def test_api_key_comes_from_process_environment(self):
        with patch.object(resume_generator.st, "secrets", {}, create=True), \
                patch.dict(os.environ, {"OPENROUTER_API_KEY": " env-test-key "}):
            self.assertEqual(resume_generator.get_api_key(), "env-test-key")

    def test_api_key_falls_back_to_streamlit_secrets(self):
        with patch.object(resume_generator.st, "secrets", {"OPENROUTER_API_KEY": "secret-test-key"}, create=True), \
                patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENROUTER_API_KEY", None)
            self.assertEqual(resume_generator.get_api_key(), "secret-test-key")

    def test_empty_environment_api_key_falls_back_to_streamlit_secrets(self):
        with patch.object(resume_generator.st, "secrets", {"OPENROUTER_API_KEY": "secret-test-key"}, create=True), \
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "   "}):
            self.assertEqual(resume_generator.get_api_key(), "secret-test-key")

    def test_environment_api_key_takes_priority_over_streamlit_secrets(self):
        with patch.object(resume_generator.st, "secrets", {"OPENROUTER_API_KEY": "secret-test-key"}, create=True), \
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-test-key"}):
            self.assertEqual(resume_generator.get_api_key(), "env-test-key")

    def test_missing_streamlit_secrets_is_safe_and_missing_key_is_controlled(self):
        missing_secrets = Mock()
        missing_secrets.get.side_effect = StreamlitSecretNotFoundError("no secrets")
        with patch.object(resume_generator.st, "secrets", missing_secrets, create=True), \
                patch.dict(os.environ, {}, clear=False), \
                patch.object(resume_generator.requests, "post") as post:
            os.environ.pop("OPENROUTER_API_KEY", None)
            self.assertIsNone(resume_generator.get_api_key())
            result = resume_generator.call_llm_raw("prompt")
        self.assertIn("OPENROUTER_API_KEY", result)
        self.assertNotIn("no secrets", result)
        post.assert_not_called()

    def test_unexpected_streamlit_secrets_error_is_not_suppressed(self):
        broken_secrets = Mock()
        broken_secrets.get.side_effect = RuntimeError("unexpected programming failure")
        with patch.object(resume_generator.st, "secrets", broken_secrets, create=True), \
                patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "unexpected programming failure"):
                resume_generator.get_api_key()

    def test_missing_key_does_not_log_or_return_a_secret_value(self):
        marker = "never-print-this-key"
        with patch.object(resume_generator.st, "secrets", {}, create=True), \
                patch.dict(os.environ, {}, clear=False), \
                self.assertLogs(resume_generator.logger, level=logging.ERROR) as captured:
            os.environ.pop("OPENROUTER_API_KEY", None)
            result = resume_generator.call_llm_raw("prompt")
        output = result + "\n" + "\n".join(captured.output)
        self.assertNotIn(marker, output)

    def test_retry_after_parses_seconds_and_http_dates(self):
        self.assertEqual(resume_generator._retry_after_seconds("2.5"), 2.5)
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self.assertEqual(
            resume_generator._retry_after_seconds("Wed, 01 Jan 2025 00:00:03 GMT", now=now),
            3,
        )
        self.assertIsNone(resume_generator._retry_after_seconds("not-a-date"))

    def test_backoff_is_exponential_jittered_and_capped(self):
        with patch.object(resume_generator.random, "uniform", return_value=0.25):
            self.assertEqual(resume_generator._backoff_delay(1), 1.25)
            self.assertEqual(resume_generator._backoff_delay(2), 2.25)
            self.assertLessEqual(resume_generator._backoff_delay(20), resume_generator.MAX_RETRY_DELAY_SECONDS)

    def test_settings_fall_back_to_environment_without_requiring_a_secret_file(self):
        with patch.object(resume_generator.st, "secrets", {}, create=True), \
                patch.dict(os.environ, {"OPENROUTER_MODEL": "test/from-env"}):
            self.assertEqual(resume_generator._setting("OPENROUTER_MODEL"), "test/from-env")

    def test_environment_model_takes_precedence_over_streamlit_secret(self):
        with patch.object(resume_generator.st, "secrets", {"OPENROUTER_MODEL": "test/from-secret"}, create=True), \
                patch.dict(os.environ, {"OPENROUTER_MODEL": "test/from-env"}):
            self.assertEqual(resume_generator._setting("OPENROUTER_MODEL"), "test/from-env")

    def test_empty_environment_model_falls_back_to_streamlit_secret(self):
        with patch.object(resume_generator.st, "secrets", {"OPENROUTER_MODEL": "test/from-secret"}, create=True), \
                patch.dict(os.environ, {"OPENROUTER_MODEL": "   "}):
            self.assertEqual(resume_generator._setting("OPENROUTER_MODEL"), "test/from-secret")

    def test_empty_secret_is_treated_as_missing(self):
        with patch.object(resume_generator.st, "secrets", {"OPENROUTER_MODEL": "  "}, create=True), \
                patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENROUTER_MODEL", None)
            self.assertEqual(resume_generator._setting("OPENROUTER_MODEL"), "")

    def test_unexpected_model_secret_error_is_not_suppressed(self):
        broken_secrets = Mock()
        broken_secrets.get.side_effect = RuntimeError("unexpected configuration failure")
        with patch.object(resume_generator.st, "secrets", broken_secrets, create=True), \
                patch.dict(os.environ, {"OPENROUTER_MODEL": ""}):
            with self.assertRaisesRegex(RuntimeError, "unexpected configuration failure"):
                resume_generator._setting("OPENROUTER_MODEL")

    def test_missing_model_configuration_is_empty_and_does_not_call_provider(self):
        missing_secrets = Mock()
        missing_secrets.get.side_effect = StreamlitSecretNotFoundError("not configured")
        with patch.object(resume_generator.st, "secrets", missing_secrets, create=True), \
                patch.dict(os.environ, {}, clear=False), \
                patch.object(resume_generator.requests, "post") as post:
            os.environ.pop("OPENROUTER_MODEL", None)
            self.assertEqual(resume_generator._setting("OPENROUTER_MODEL"), "")
            post.assert_not_called()

    def test_configuration_lookup_error_returns_sanitized_user_message(self):
        secret_marker = "never-display-this-configuration-detail"
        broken_secrets = Mock()
        broken_secrets.get.side_effect = RuntimeError(secret_marker)
        with patch.object(resume_generator.st, "secrets", broken_secrets, create=True), \
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": ""}), \
                self.assertLogs(resume_generator.logger, level=logging.ERROR) as captured:
            result = resume_generator.call_llm("prompt")
        output = result + "\n" + "\n".join(captured.output)
        self.assertIn("configuration could not be read", result)
        self.assertNotIn(secret_marker, output)
        self.assertNotIn("test-key", output)

    def test_expired_total_budget_prevents_a_request(self):
        with patch.object(resume_generator.time, "monotonic", side_effect=[10.0, 10.0]), \
                patch.object(resume_generator.requests, "post") as post:
            content, error = resume_generator._request_model("prompt", "test/model", "test-key", deadline=10.0)
        self.assertIsNone(content)
        self.assertEqual(error, "timeout")
        post.assert_not_called()

    def test_response_received_after_deadline_is_discarded(self):
        response = make_response(200, {"choices": [{"message": {"content": "too late"}}]})
        with patch.object(resume_generator.time, "monotonic", side_effect=[0.0, 0.0, 10.1]), \
                patch.object(resume_generator.requests, "post", return_value=response) as post:
            content, error = resume_generator._request_model("prompt", "test/model", "test-key", deadline=10.0)
        self.assertIsNone(content)
        self.assertEqual(error, "timeout")
        post.assert_called_once()
        response.json.assert_not_called()

    def test_retry_does_not_start_when_delay_exceeds_remaining_budget(self):
        response = make_response(429, headers={"Retry-After": "1"})
        with patch.object(resume_generator.time, "monotonic", side_effect=[0.0, 0.0, 0.1, 0.1]), \
                patch.object(resume_generator.requests, "post", return_value=response) as post, \
                patch.object(resume_generator.time, "sleep") as sleep:
            content, error = resume_generator._request_model("prompt", "test/model", "test-key", deadline=0.5)
        self.assertIsNone(content)
        self.assertEqual(error, "rate_limited")
        post.assert_called_once()
        sleep.assert_not_called()

    def test_rate_limit_retries_three_times_and_uses_retry_after(self):
        limited = make_response(429, headers={"Retry-After": "2"})
        success = make_response(200, {"choices": [{"message": {"content": "ok"}}]})
        with patch.object(resume_generator.requests, "post", side_effect=[limited, limited, success]) as post, \
                patch.object(resume_generator.time, "sleep") as sleep:
            content, error = resume_generator._request_model("prompt", "test/model", "test-key")
        self.assertEqual((content, error), ("ok", None))
        self.assertEqual(post.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(2.0), call(2.0)])

    def test_http_date_retry_after_is_used_for_mocked_rate_limit(self):
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        retry_at = format_datetime(now + timedelta(seconds=5), usegmt=True)
        limited = make_response(429, headers={"Retry-After": retry_at})
        success = make_response(200, {"choices": [{"message": {"content": "ok"}}]})
        with patch.object(resume_generator, "datetime") as clock, \
                patch.object(resume_generator.time, "monotonic", return_value=0.0), \
                patch.object(resume_generator.requests, "post", side_effect=[limited, success]) as post, \
                patch.object(resume_generator.time, "sleep") as sleep:
            clock.now.return_value = now
            content, error = resume_generator._request_model("prompt", "test/model", "test-key", deadline=30.0)
        self.assertEqual((content, error), ("ok", None))
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(5.0)

    def test_retry_after_above_maximum_stops_without_sleeping(self):
        response = make_response(429, headers={"Retry-After": "3600"})
        with patch.object(resume_generator.requests, "post", return_value=response) as post, \
                patch.object(resume_generator.time, "sleep") as sleep:
            content, error = resume_generator._request_model("prompt", "test/model", "test-key")
        self.assertIsNone(content)
        self.assertEqual(error, "rate_limited")
        post.assert_called_once()
        sleep.assert_not_called()

    def test_timeout_retries_are_bounded_without_final_sleep(self):
        with patch.object(
            resume_generator.requests,
            "post",
            side_effect=resume_generator.requests.exceptions.Timeout("private prompt marker"),
        ) as post, patch.object(resume_generator.time, "sleep") as sleep:
            content, error = resume_generator._request_model("private prompt marker", "test/model", "test-key")
        self.assertIsNone(content)
        self.assertEqual(error, "timeout")
        self.assertEqual(post.call_count, resume_generator.MAX_RETRIES)
        self.assertEqual(sleep.call_count, resume_generator.MAX_RETRIES - 1)

    def test_server_errors_retry_but_remain_bounded(self):
        response = make_response(503)
        with patch.object(resume_generator.requests, "post", return_value=response) as post, \
                patch.object(resume_generator.time, "sleep") as sleep, \
                patch.object(resume_generator.random, "uniform", return_value=0):
            content, error = resume_generator._request_model("prompt", "test/model", "test-key")
        self.assertIsNone(content)
        self.assertEqual(error, "service_unavailable")
        self.assertEqual(post.call_count, resume_generator.MAX_RETRIES)
        self.assertEqual(sleep.call_count, resume_generator.MAX_RETRIES - 1)

    def test_timeout_and_server_statuses_have_consistent_categories(self):
        for status, expected in (
            (408, "timeout"),
            (500, "service_unavailable"),
            (502, "service_unavailable"),
            (503, "service_unavailable"),
            (504, "service_unavailable"),
        ):
            with self.subTest(status=status):
                self.assertEqual(resume_generator._error_category(status), expected)

    def test_invalid_success_body_retries_without_logging_response_text(self):
        response = make_response(200)
        response.json.side_effect = ValueError("private provider response")
        with patch.object(resume_generator.requests, "post", return_value=response) as post, \
                patch.object(resume_generator.time, "sleep"), \
                patch.object(resume_generator.random, "uniform", return_value=0):
            content, error = resume_generator._request_model("private prompt", "test/model", "test-key")
        self.assertIsNone(content)
        self.assertEqual(error, "invalid_response")
        self.assertEqual(post.call_count, resume_generator.MAX_RETRIES)

    def test_empty_model_completion_retries_then_returns_controlled_error(self):
        response = make_response(200, {"choices": [{"message": {"content": "  "}}]})
        with patch.object(resume_generator.requests, "post", return_value=response) as post, \
                patch.object(resume_generator.time, "sleep"):
            content, error = resume_generator._request_model("prompt", "test/model", "test-key")
        self.assertIsNone(content)
        self.assertEqual(error, "empty_response")
        self.assertEqual(post.call_count, resume_generator.MAX_RETRIES)

    def test_timeout_error_message_never_contains_api_key(self):
        api_key = "never-display-this-api-key"
        configuration = ("test/model", "", "http://localhost:8501", "Test App")
        with patch.object(resume_generator, "get_api_key", return_value=api_key), \
                patch.object(resume_generator, "_generation_configuration", return_value=configuration), \
                patch.object(resume_generator, "_request_model", return_value=(None, "timeout")):
            result = resume_generator.call_llm_raw("prompt")
        self.assertIn("timed out", result)
        self.assertNotIn(api_key, result)

    def test_public_call_llm_prompt_signature_remains_compatible(self):
        configuration = ("test/model", "", "http://localhost:8501", "AI Career Platform")
        with patch.object(resume_generator, "_generation_configuration", return_value=configuration), \
                patch.object(resume_generator, "_cached_call_llm", return_value="generated") as cached:
            self.assertEqual(resume_generator.call_llm("prompt"), "generated")
        cached.assert_called_once_with("prompt", False, configuration)

    def test_connection_and_network_errors_have_sanitized_categories(self):
        exceptions = (
            (resume_generator.requests.exceptions.ConnectionError("secret detail"), "connection_error"),
            (resume_generator.requests.exceptions.RequestException("secret detail"), "network_error"),
        )
        for exception, expected in exceptions:
            with self.subTest(expected=expected), \
                    patch.object(resume_generator.requests, "post", side_effect=exception), \
                    patch.object(resume_generator.time, "sleep"):
                content, error = resume_generator._request_model("prompt", "test/model", "test-key")
            self.assertIsNone(content)
            self.assertEqual(error, expected)

    def test_unavailable_primary_does_not_fallback_without_explicit_model(self):
        settings = {"OPENROUTER_MODEL": "test/primary", "OPENROUTER_FALLBACK_MODEL": ""}
        with patch.object(resume_generator, "get_api_key", return_value="test-key"), \
                patch.object(resume_generator, "_setting", side_effect=lambda name, default="": settings.get(name, default)), \
                patch.object(resume_generator, "_request_model", return_value=(None, "model_unavailable")) as request:
            result = resume_generator.call_llm_raw("prompt")
        self.assertIn("model is unavailable", result)
        request.assert_called_once()

    def test_primary_and_fallback_share_the_total_request_deadline(self):
        settings = {"OPENROUTER_MODEL": "test/primary", "OPENROUTER_FALLBACK_MODEL": "test/fallback"}
        with patch.object(resume_generator, "get_api_key", return_value="test-key"), \
                patch.object(resume_generator, "_setting", side_effect=lambda name, default="": settings.get(name, default)), \
                patch.object(resume_generator.time, "monotonic", return_value=100.0), \
                patch.object(
                    resume_generator,
                    "_request_model",
                    side_effect=[(None, "model_unavailable"), (None, "timeout")],
                ) as request:
            resume_generator.call_llm_raw("prompt")
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[0].args[3], 145.0)
        self.assertEqual(request.call_args_list[1].args[3], 145.0)

    def test_unavailable_or_access_denied_does_not_retry_or_fallback_unnecessarily(self):
        for category in ("invalid_request", "access_denied", "service_unavailable"):
            with self.subTest(category=category):
                settings = {"OPENROUTER_MODEL": "test/primary", "OPENROUTER_FALLBACK_MODEL": "test/fallback"}
                with patch.object(resume_generator, "get_api_key", return_value="test-key"), \
                        patch.object(resume_generator, "_setting", side_effect=lambda name, default="": settings.get(name, default)), \
                        patch.object(resume_generator, "_request_model", return_value=(None, category)) as request:
                    resume_generator.call_llm_raw("prompt")
                request.assert_called_once()

    def test_model_configuration_changes_invalidate_cached_output_without_key_in_cache_identity(self):
        resume_generator._cached_call_llm.clear()
        self.addCleanup(resume_generator._cached_call_llm.clear)
        settings = {
            "OPENROUTER_MODEL": "test/primary-one",
            "OPENROUTER_FALLBACK_MODEL": "test/fallback-one",
            "OPENROUTER_SITE_URL": "https://example.test",
            "OPENROUTER_APP_NAME": "Test App",
        }
        responses = [
            make_response(200, {"choices": [{"message": {"content": result}}]})
            for result in ("first", "second", "third")
        ]
        with patch.object(resume_generator, "get_api_key", return_value="test-api-key"), \
                patch.object(resume_generator, "_setting", side_effect=lambda name, default="": settings.get(name, default)), \
                patch.object(resume_generator.requests, "post", side_effect=responses) as post:
            first = resume_generator.call_llm("same prompt")
            self.assertEqual(resume_generator.call_llm("same prompt"), first)
            settings["OPENROUTER_MODEL"] = "test/primary-two"
            second = resume_generator.call_llm("same prompt")
            settings["OPENROUTER_FALLBACK_MODEL"] = "test/fallback-two"
            third = resume_generator.call_llm("same prompt")

        self.assertEqual((first, second, third), ("first", "second", "third"))
        self.assertEqual(post.call_count, 3)
        self.assertEqual(
            [entry.kwargs["json"]["model"] for entry in post.call_args_list],
            ["test/primary-one", "test/primary-two", "test/primary-two"],
        )
        self.assertNotIn("test-api-key", repr(resume_generator._generation_configuration()))

    def test_http_errors_are_classified_and_auth_errors_stop_immediately(self):
        cases = {
            400: "invalid_request",
            401: "authentication_error",
            402: "insufficient_credits",
            403: "access_denied",
            404: "model_unavailable",
        }
        for status, expected in cases.items():
            with self.subTest(status=status):
                response = make_response(status)
                with patch.object(resume_generator.requests, "post", return_value=response) as post:
                    content, error = resume_generator._request_model("prompt", "test/model", "test-key")
                self.assertIsNone(content)
                self.assertEqual(error, expected)
                post.assert_called_once()

    def test_model_fallback_requires_configuration_and_only_follows_unavailable(self):
        settings = {"OPENROUTER_MODEL": "test/primary", "OPENROUTER_FALLBACK_MODEL": "test/fallback"}
        with patch.object(resume_generator, "get_api_key", return_value="test-key"), \
                patch.object(resume_generator, "_setting", side_effect=lambda name, default="": settings.get(name, default)), \
                patch.object(
                    resume_generator,
                    "_request_model",
                    side_effect=[(None, "model_unavailable"), ("fallback output", None)],
                ) as request:
            result = resume_generator.call_llm_raw("prompt")
        self.assertEqual(result, "fallback output")
        self.assertEqual([item.args[1] for item in request.call_args_list], ["test/primary", "test/fallback"])

    def test_no_fallback_is_attempted_for_authentication_or_credits(self):
        for category in ("authentication_error", "insufficient_credits"):
            with self.subTest(category=category):
                settings = {"OPENROUTER_MODEL": "test/primary", "OPENROUTER_FALLBACK_MODEL": "test/fallback"}
                with patch.object(resume_generator, "get_api_key", return_value="test-key"), \
                        patch.object(resume_generator, "_setting", side_effect=lambda name, default="": settings.get(name, default)), \
                        patch.object(resume_generator, "_request_model", return_value=(None, category)) as request:
                    result = resume_generator.call_llm_raw("prompt")
                self.assertTrue(result.startswith("⚠️"))
                request.assert_called_once()

    def test_model_configuration_is_required_and_never_defaults_to_a_model(self):
        with patch.object(resume_generator, "get_api_key", return_value="test-key"), \
                patch.object(resume_generator, "_setting", return_value=""), \
                patch.object(resume_generator.requests, "post") as post:
            result = resume_generator.call_llm_raw("prompt")
        self.assertIn("OPENROUTER_MODEL", result)
        post.assert_not_called()

    def test_diagnostics_do_not_include_prompt_key_or_provider_body(self):
        prompt = "candidate-sensitive-prompt-731"
        api_key = "api-secret-test-value"
        provider_text = "provider-echoed-sensitive-body-942"
        response = make_response(500, {"error": {"message": provider_text}})
        with self.assertLogs(resume_generator.logger, level=logging.WARNING) as captured, \
                patch.object(resume_generator.requests, "post", return_value=response), \
                patch.object(resume_generator.time, "sleep"):
            resume_generator._request_model(prompt, "test/model", api_key)
        output = "\n".join(captured.output)
        for sensitive in (prompt, api_key, provider_text):
            self.assertNotIn(sensitive, output)


class PortfolioAndATSRegressionTests(unittest.TestCase):
    def test_resume_and_cover_letter_generation_keep_existing_call_contracts(self):
        profile = {"name": "Candidate", "target_role": "Engineer"}
        with patch.object(resume_generator, "call_llm", return_value="generated") as call_llm:
            self.assertEqual(resume_generator.generate_resume(profile), "generated")
            self.assertEqual(resume_generator.generate_cover_letter(profile), "generated")
        self.assertEqual(call_llm.call_count, 2)

    def test_json_extraction_handles_fences_nested_objects_and_quoted_braces(self):
        response = r'''Model result:
```json
{
            "about": "Built systems with {braces} and escaped \"quotes\"",
            "projects": [{"title": "API {v2}", "description": "Handled } in a string and nested {data: true}"}]
        }
```
'''
        extracted = resume_generator.extract_json(response)
        data = resume_generator._validate_portfolio_data(json.loads(extracted))
        self.assertEqual(data["about"], 'Built systems with {braces} and escaped "quotes"')
        self.assertEqual(data["projects"][0]["title"], "API {v2}")

    def test_malformed_portfolio_json_returns_a_friendly_error(self):
        malformed = '```json\n{"about": "unfinished", "projects": [}\n```'
        with patch.object(resume_generator, "call_llm", return_value=malformed):
            result = resume_generator.generate_portfolio_data({"name": "Candidate"})
        self.assertTrue(result["about"].startswith("⚠️"))
        self.assertEqual(result["projects"], [])

    def test_portfolio_schema_rejects_wrong_required_fields_and_project_entries(self):
        invalid_values = [
            {"about": "Engineer"},
            {"about": [], "projects": []},
            {"about": "Engineer", "projects": ["not an object"]},
            {"about": "Engineer", "projects": [{"title": "API", "description": 7}]},
        ]
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                resume_generator._validate_portfolio_data(value)

    def test_portfolio_failure_is_friendly_and_does_not_log_response_content(self):
        raw = 'unparsed private response 127: {"about": []}'
        with self.assertLogs(resume_generator.logger, level=logging.WARNING) as captured, \
                patch.object(resume_generator, "call_llm", return_value=raw):
            result = resume_generator.generate_portfolio_data({"name": "Candidate"})
        self.assertTrue(result["about"].startswith("⚠️"))
        self.assertEqual(result["projects"], [])
        self.assertNotIn(raw, "\n".join(captured.output))

    def test_portfolio_prompt_formats_json_example_and_both_templates_render(self):
        response = '{"about":"Engineer","projects":[{"title":"API","description":"Built an API"}]}'
        with patch.object(resume_generator, "call_llm", return_value=response) as call_llm:
            data = resume_generator.generate_portfolio_data({"name": "Candidate", "projects": "API"})
        prompt = call_llm.call_args.args[0]
        self.assertIn('"projects": [', prompt)
        self.assertNotIn("{{", prompt)
        for theme in ("portfolio_minimal", "portfolio_modern"):
            html = build_portfolio_html(data, theme, {"name": "Candidate"})
            self.assertIn("Engineer", html)
            self.assertIn("API", html)

    def test_ats_parses_valid_json_and_returns_safe_failure_for_malformed_output(self):
        valid = '{"match_score": 82, "missing_skills": ["Docker"], "suggestions": ["Add Docker"]}'
        with patch.object(similarity, "call_llm", return_value=valid):
            result = similarity.analyze_ats_match("Python developer", "Python role")
        self.assertEqual(result["match_score"], 82)
        self.assertEqual(result["missing_skills"], ["Docker"])

        secretish = "malformed ATS response with private text 418"
        with self.assertLogs(similarity.logger, level=logging.WARNING) as captured, \
                patch.object(similarity, "call_llm", return_value=secretish):
            failure = similarity.analyze_ats_match("Python developer", "Python role")
        self.assertIsNone(failure["match_score"])
        self.assertIn("error", failure)
        self.assertNotIn(secretish, "\n".join(captured.output))
        self.assertTrue(all(secretish not in suggestion for suggestion in failure["suggestions"]))

    def test_ats_rejects_missing_fields_wrong_types_booleans_and_out_of_range_scores(self):
        invalid_responses = (
            '{"match_score": 82, "missing_skills": ["Docker"]}',
            '{"match_score": "82", "missing_skills": [], "suggestions": []}',
            '{"match_score": true, "missing_skills": [], "suggestions": []}',
            '{"match_score": -1, "missing_skills": [], "suggestions": []}',
            '{"match_score": 101, "missing_skills": [], "suggestions": []}',
            '{"match_score": 50, "missing_skills": "Docker", "suggestions": []}',
            '{"match_score": 50, "missing_skills": [], "suggestions": {}}',
        )
        for response in invalid_responses:
            with self.subTest(response=response), patch.object(similarity, "call_llm", return_value=response):
                result = similarity.analyze_ats_match("profile", "job description")
            self.assertIsNone(result["match_score"])
            self.assertTrue(result["error"])


class ApplicationBehaviorTests(unittest.TestCase):
    def _run_app_generation_button(self, button_fragment, patched_generator, initial_state=None):
        with patched_generator:
            app_test = AppTest.from_file(str(Path(__file__).with_name("app.py")))
            app_test.session_state["name"] = "Synthetic Candidate"
            app_test.session_state["target_role"] = "Software Engineer"
            app_test.session_state["skills"] = "Python"
            app_test.session_state["projects"] = "Sample project"
            app_test.session_state["job_description"] = "Python software engineer"
            for key, value in (initial_state or {}).items():
                app_test.session_state[key] = value
            app_test.run()
            button = next(item for item in app_test.button if button_fragment in item.label)
            button.click().run()
        return app_test

    def test_resume_generation_error_is_displayed_without_sensitive_text(self):
        secret_marker = "private-provider-token"
        app_test = self._run_app_generation_button(
            "Generate / Regenerate Base Resume",
            patch.object(
                resume_generator,
                "generate_resume",
                return_value=f"⚠️ provider error {secret_marker}",
            ),
            initial_state={
                "resume_text": "previous resume",
                "resume_versions": ["previous resume"],
                "live_resume_editor": "previous resume",
            },
        )
        errors = " ".join(element.value for element in app_test.error)
        self.assertIn("Resume generation could not be completed", errors)
        self.assertNotIn(secret_marker, errors)
        self.assertEqual(app_test.session_state["resume_text"], "")
        self.assertEqual(app_test.session_state["resume_versions"], [])
        self.assertNotIn("live_resume_editor", app_test.session_state)
        self.assertEqual(len(app_test.exception), 0)

    def test_successful_resume_regeneration_replaces_previous_content(self):
        app_test = self._run_app_generation_button(
            "Generate / Regenerate Base Resume",
            patch.object(resume_generator, "generate_resume", return_value="new resume"),
            initial_state={
                "resume_text": "previous resume",
                "resume_versions": ["previous resume"],
                "live_resume_editor": "previous resume",
            },
        )
        self.assertEqual(app_test.session_state["resume_text"], "new resume")
        self.assertEqual(app_test.session_state["resume_versions"], ["new resume"])
        self.assertEqual(app_test.session_state["live_resume_editor"], "new resume")
        self.assertEqual(len(app_test.exception), 0)

    def test_cover_letter_generation_error_is_displayed(self):
        app_test = self._run_app_generation_button(
            "Generate / Regenerate Base Cover Letter",
            patch.object(
                resume_generator,
                "generate_cover_letter",
                return_value="⚠️ provider error",
            ),
            initial_state={
                "cover_letter_text": "previous letter",
                "cover_letter_versions": ["previous letter"],
                "live_cl_editor": "previous letter",
            },
        )
        self.assertTrue(any("Cover letter generation could not be completed" in e.value for e in app_test.error))
        self.assertEqual(app_test.session_state["cover_letter_text"], "")
        self.assertEqual(app_test.session_state["cover_letter_versions"], [])
        self.assertNotIn("live_cl_editor", app_test.session_state)
        self.assertEqual(len(app_test.exception), 0)

    def test_successful_cover_letter_regeneration_replaces_previous_content(self):
        app_test = self._run_app_generation_button(
            "Generate / Regenerate Base Cover Letter",
            patch.object(resume_generator, "generate_cover_letter", return_value="new letter"),
            initial_state={
                "cover_letter_text": "previous letter",
                "cover_letter_versions": ["previous letter"],
                "live_cl_editor": "previous letter",
            },
        )
        self.assertEqual(app_test.session_state["cover_letter_text"], "new letter")
        self.assertEqual(app_test.session_state["cover_letter_versions"], ["new letter"])
        self.assertEqual(app_test.session_state["live_cl_editor"], "new letter")
        self.assertEqual(len(app_test.exception), 0)

    def test_non_string_cover_letter_refinement_shows_safe_error(self):
        app_test = self._run_app_generation_button(
            "Shorter",
            patch.object(resume_generator, "refine_cover_letter_tone", return_value=None),
            initial_state={
                "cover_letter_text": "existing letter",
                "cover_letter_versions": ["existing letter"],
                "live_cl_editor": "existing letter",
            },
        )
        self.assertTrue(any("Cover letter generation returned an invalid result" in e.value for e in app_test.error))
        self.assertEqual(app_test.session_state["cover_letter_text"], "existing letter")
        self.assertEqual(len(app_test.exception), 0)

    def test_portfolio_generation_error_is_displayed(self):
        app_test = self._run_app_generation_button(
            "Generate / Preview Live Portfolio",
            patch.object(
                resume_generator,
                "generate_portfolio_data",
                return_value={"about": "⚠️ provider error", "projects": []},
            ),
        )
        self.assertTrue(any("Portfolio generation could not be completed" in e.value for e in app_test.error))
        self.assertEqual(len(app_test.exception), 0)

    def test_ats_failure_is_displayed_as_a_controlled_error(self):
        secret_marker = "private-provider-token"
        app_test = self._run_app_generation_button(
            "Calculate / Recalculate ATS Match Score",
            patch.object(
                similarity,
                "analyze_ats_match",
                return_value={"match_score": None, "error": secret_marker},
            ),
        )
        errors = " ".join(element.value for element in app_test.error)
        self.assertIn("ATS analysis could not be completed", errors)
        self.assertNotIn(secret_marker, errors)
        self.assertEqual(len(app_test.exception), 0)

    def test_version_restore_updates_document_and_editor_state(self):
        for text_key, versions_key, editor_key in (
            ("resume_text", "resume_versions", "live_resume_editor"),
            ("cover_letter_text", "cover_letter_versions", "live_cl_editor"),
        ):
            with self.subTest(text_key=text_key):
                state = {
                    text_key: "current version",
                    versions_key: ["restored version", "current version"],
                    editor_key: "current version",
                }
                restored = streamlit_app._restore_document_version(
                    state, text_key, versions_key, editor_key, 0
                )
                self.assertTrue(restored)
                self.assertEqual(state[text_key], "restored version")
                self.assertEqual(state[editor_key], "restored version")

    def test_ats_error_renders_an_error_instead_of_a_zero_score(self):
        with patch.object(streamlit_app.st, "error") as show_error, \
                patch.object(streamlit_app.st, "progress") as show_progress, \
                patch.object(streamlit_app.st, "success") as show_success, \
                patch.object(streamlit_app.st, "warning") as show_warning:
            streamlit_app._render_ats_result({"match_score": None, "error": "private-provider-token"})
        show_error.assert_called_once_with(
            "ATS analysis could not be completed. Please review your inputs and try again."
        )
        show_progress.assert_not_called()
        show_success.assert_not_called()
        show_warning.assert_not_called()


class DocumentExportAndUIRegressionTests(unittest.TestCase):
    def test_pdf_and_docx_contracts_return_in_memory_bytes(self):
        markdown = "# Candidate\n\n## Skills\n- Python"
        pdf = pdf_generator.compile_pdf(markdown)
        docx = docx_generator.compile_docx(markdown)
        self.assertIsInstance(pdf, bytes)
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertIsInstance(docx, bytes)
        self.assertTrue(docx.startswith(b"PK"))

    def test_pdf_invalid_input_and_encoding_failures_are_controlled(self):
        for value in (None, "", "   "):
            with self.subTest(value=value), self.assertRaises(pdf_generator.PDFGenerationError):
                pdf_generator.compile_pdf(value)
        with self.assertRaises(pdf_generator.PDFGenerationError):
            pdf_generator.compile_pdf("x" * (pdf_generator.MAX_MARKDOWN_LENGTH + 1))
        with self.assertRaises(pdf_generator.PDFGenerationError):
            pdf_generator.compile_pdf("# Candidate\n\nEmoji: 🧪")
        with patch.object(pdf_generator, "ATSResumePDF", side_effect=RuntimeError("internal detail")):
            with self.assertRaises(pdf_generator.PDFGenerationError):
                pdf_generator.compile_pdf("# Candidate")

if __name__ == "__main__":
    unittest.main()
