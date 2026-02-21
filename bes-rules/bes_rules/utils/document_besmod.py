import asyncio
import logging
import os
from pathlib import Path

from poe_api_wrapper import AsyncPoeApi
from bes_rules import STARTUP_BESMOD_MOS

from bes_rules.utils.process_papers import get_tokens

logger = logging.getLogger(__name__)


def find_undocumented_mo_files(root_dir):
    undocumented_files = {}

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if not file.endswith('.mo'):
                continue
            if (
                    file.endswith("package.mo") or
                    file.endswith("Bus.mo") or
                    file.endswith("SingleDwelling.mo") or
                    file.endswith("Icon.mo") or
                    file.endswith("Outputs.mo")
            ):
                continue
            file_path = os.path.join(root, file)
            if "TimeConstantEstimation" in file_path or "OpenModelicaErrors" in file_path:
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'Documentation' not in content:
                    undocumented_files[Path(file_path)] = content
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")

    return undocumented_files


def get_prompt():
    return """
Please create a concise html documentation for the Modelica below. 
Instructions:
- Link relevant related models only like this: <a href=\"modelica://PATH.TO.MODEL\">PATH.TO.MODEL</a>
- Use <h4> for section headers, and include at least the section "Information". 
- If you think it is relevant, also add a section "Important Parameters"
- If references are stated anywhere, include a section "References" 
- Do not include simulation settings
- Only use utf8 encoding
- Only respond with the html code!

Model:
"""


async def main(file_path, model_code):
    bot_name = "claude_3_sonnet_200k"
    client = await AsyncPoeApi(tokens=get_tokens()).create()
    message = get_prompt() + "\n\n" + model_code
    response = ""
    try:
        async for chunk in client.send_message(bot=bot_name, message=message):
            response += chunk["response"]
            print(chunk["response"], end="", flush=True)
    except RuntimeError as err:
        logger.error("TimeoutError lead to RunTime error, response should be there anyway: %s")

    response = response.replace("```html", "").replace("```", "")
    response = response.replace('"', '\\"')
    write_documentation_to_file(file_path=file_path, documentation=response)


def run_asyncio(file_path, model_code):
    asyncio.run(main(file_path, model_code))


def document_besmod():
    undocumented_files = find_undocumented_mo_files(STARTUP_BESMOD_MOS.parent.joinpath("BESMod"))
    import multiprocessing as mp
    for file_path, model_code in undocumented_files.items():
        proc = mp.Process(target=run_asyncio, args=(file_path, model_code))
        proc.start()
        proc.join()


def write_documentation_to_file(file_path: Path, documentation):
    if not documentation:
        return

    def get_annotation(doc):
        return f'  annotation (Documentation(info="<html>{doc}</html>"));\n'

    with open(file_path, "r") as file:
        contents = file.readlines()
    contents_new = contents.copy()
    idx = 0
    for idx, line in enumerate(contents[::-1]):
        if line.startswith("end "):
            break
    contents_new.insert(len(contents_new) - idx - 1, get_annotation(documentation))

    with open(file_path, "w") as file:
        file.write("".join(contents_new))


if __name__ == '__main__':
    d = find_undocumented_mo_files(STARTUP_BESMOD_MOS.parent.joinpath("BESMod"))
    document_besmod()
