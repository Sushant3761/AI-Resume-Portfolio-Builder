import os
import re
from jinja2 import Environment, FileSystemLoader, select_autoescape

def build_portfolio_html(json_data: dict, template_name: str, applicant_data: dict) -> str:
    """
    Renders structured LLM JSON data into an elegant HTML template using Jinja2.
    """
    # 1. Setup Jinja Environment
    # We load templates from the templates/ directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates_dir = os.path.join(base_dir, "templates")
    
    # If the directory doesn't exist, this will crash. We assume the templates exist per the plan.
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    
    # 2. Get the requested template wrapper
    try:
        template = env.get_template(f"{template_name}.html")
    except Exception:
        # Fallback to minimal if requested doesn't exist
        template = env.get_template("portfolio_minimal.html")
        
    # 3. Compile variables for Jinja injection
    # Extract robust defaults, passing our secure json dictionary
    template_vars = {
        "name": applicant_data.get('name', 'Developer Persona'),
        "target_role": applicant_data.get('target_role', 'Software Engineer'),
        "skills": applicant_data.get('skills', 'No skills specified'),
        "contact": "Contact info not specified", # can be extended
        "about": json_data.get("about", "Not specified"),
        "projects": json_data.get("projects", [])
    }
    
    # 4. Render
    raw_html = template.render(**template_vars)
    return raw_html
