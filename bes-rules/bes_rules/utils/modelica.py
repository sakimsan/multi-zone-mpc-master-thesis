import pathlib


def load_modelica_file_modifier(file: pathlib.Path):
    file = str(file).replace("\\", "//")
    return f'Modelica.Utilities.Files.loadResource("{file}")'
