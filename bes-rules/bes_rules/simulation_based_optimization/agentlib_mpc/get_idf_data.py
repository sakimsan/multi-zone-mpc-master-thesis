import pandas as pd
import numpy as np
from pathlib import Path
from bes_rules import RESULTS_FOLDER

def polygon_area_3d(coords):
    if len(coords) < 3:
        return 0.0
    area_vector = np.array([0.0, 0.0, 0.0])
    for i in range(len(coords)):
        p1 = np.array(coords[i])
        p2 = np.array(coords[(i + 1) % len(coords)])
        area_vector += np.cross(p1, p2)
    return 0.5 * np.linalg.norm(area_vector)

def parse_surface_blocks(lines):
    surfaces = []
    current_block = []
    in_surface = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("buildingsurface:detailed"):
            in_surface = True
            current_block = [stripped]
            continue
        if in_surface:
            current_block.append(stripped)
            if ";" in stripped:
                surfaces.append(current_block)
                in_surface = False
    return surfaces

def extract_surface_data(surfaces, target_types, label_map=None):
    def get_height(coords):
        z_vals = [pt[2] for pt in coords]
        return round(max(z_vals) - min(z_vals), 6) if z_vals else 0.0
    entries = []
    for block in surfaces:
        data = [l.strip(" ;,\n") for l in block if l.strip()]
        if len(data) < 11:
            continue
        try:
            name = data[1].split("!-")[0].strip()
            surface_type = data[2].split("!-")[0].strip().strip(",").lower()
            construction_name = data[3].split("!-")[0].strip()
            zone_name = data[4].split("!-")[0].strip()
        except IndexError:
            continue
        # Mapping auf gewünschte Label (z.B. Ceiling → Decke)
        surface_label = label_map.get(surface_type, surface_type) if label_map else surface_type
        # Differenzierung Boden:
        if surface_label == "floor":
            if "groundfloor" in construction_name.lower():
                surface_label = "GroundFloor"
            else:
                surface_label = "InnerFloor"
        if surface_label not in target_types:
            continue
        start_index = None
        for i, line in enumerate(data):
            if "Number of Vertices" in line:
                start_index = i + 1
                break
        if start_index is None:
            continue
        coord_lines = data[start_index:]
        coords = []
        for line in coord_lines:
            coord_part = line.split("!")[0].strip(" ,;\n")
            try:
                xyz = [float(v.strip()) for v in coord_part.split(",") if v.strip()]
                if len(xyz) == 3:
                    coords.append(tuple(xyz))
            except ValueError:
                continue
        if len(coords) < 3:
            continue
        height = get_height(coords)
        area = polygon_area_3d(coords)
        # Filterlogik (wie gehabt):
        if surface_label == "wall" and height <= 0:
            continue
        if surface_label in ["boden", "decke"] and height > 0:
            continue
        # Roof-Flächen: keine Einschränkung
        entries.append({
            "zone": zone_name.split(",")[0].strip(),
            "surface_type": surface_label,
            "surface_name": name,
            "construction": construction_name,
            "area": round(area, 3)
        })
    return pd.DataFrame(entries)

def create_multicolumn_table(df, group_label):
    grouped = df.groupby("zone")
    rows = []
    for zone, group in grouped:
        row = {"zone": zone}
        for i, (_, entry) in enumerate(group.iterrows(), start=1):
            row[f"{group_label}{i if i > 1 else ''}"] = entry["area"]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("zone").fillna(0)

def get_wall_surface_mapping(lines):
    surfaces = parse_surface_blocks(lines)
    wall_entries = []
    for block in surfaces:
        data = [l.strip(" ;,\n") for l in block if l.strip()]
        if len(data) < 5:
            continue
        try:
            name = data[1].split("!-")[0].strip()
            surface_type = data[2].split("!-")[0].strip().strip(",").lower()
            zone_name = data[4].split("!-")[0].strip()
        except IndexError:
            continue
        if surface_type != "wall":
            continue
        wall_entries.append({
            "surface_name": name,
            "zone": zone_name.split(",")[0].strip()
        })
    wall_df = pd.DataFrame(wall_entries)
    wall_df = wall_df.sort_values(by=["zone", "surface_name"])
    wall_df["wand_nummer"] = wall_df.groupby("zone").cumcount() + 1
    wall_df["wand_label"] = wall_df["wand_nummer"].apply(lambda x: f"Wand{x}")
    return wall_df.set_index("surface_name")

def get_exact_window_data(lines):
    blocks = []
    current_block = []
    in_window = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("fenestrationsurface:detailed"):
            in_window = True
            current_block = [stripped]
            continue
        if in_window:
            current_block.append(stripped)
            if ";" in stripped:
                blocks.append(current_block)
                in_window = False
    entries = []
    for block in blocks:
        data = [l.strip(" ;,\n") for l in block if l.strip()]
        if len(data) < 11:
            continue
        try:
            surface_type = data[2].split("!-")[0].strip().strip(",").lower()
            if surface_type != "window":
                continue
            name = data[1].split("!-")[0].strip()
            building_surface = data[4].split("!-")[0].strip()
        except IndexError:
            continue
        start_index = None
        for i, line in enumerate(data):
            if "Number of Vertices" in line:
                start_index = i + 1
                break
        if start_index is None:
            continue
        coord_lines = data[start_index:]
        coords = []
        for line in coord_lines:
            coord_part = line.split("!")[0].strip(" ,;\n")
            try:
                xyz = [float(v.strip()) for v in coord_part.split(",") if v.strip()]
                if len(xyz) == 3:
                    coords.append(tuple(xyz))
            except ValueError:
                continue
        if len(coords) < 3:
            continue
        area = polygon_area_3d(coords)
        entries.append({
            "window_name": name,
            "building_surface": building_surface,
            "area": round(area, 3)
        })
    return pd.DataFrame(entries)

# Kapazität und Wiederstand werden bei Innerwall und Innerfloor durch 2 geteilt

def get_material_and_construction_data(lines):

    idf_parameter : bool = True


    def remove_comment(line):
        return line.split("!-")[0].strip(" ,;\n").strip('"').strip("'").lower()

    def should_halve(construction_name: str) -> bool:
        s = construction_name.lower()
        return ("bps-innerfloor" in s) or ("bps-innerwall" in s)

    # --- Material (massive Schichten) ---
    materials = {}
    in_material = False
    current = []
    for line in lines:
        line = line.strip()
        if line.startswith("Material,"):
            in_material = True
            current = []
            continue
        if in_material:
            current.append(remove_comment(line))
            if ";" in line:
                if len(current) >= 6:
                    name = current[0]
                    try:
                        thickness = float(current[2])
                        conductivity = float(current[3])
                        density = float(current[4])
                        specific_heat = float(current[5])
                        materials[name] = {
                            "thickness": thickness,
                            "conductivity": conductivity,
                            "density": density,
                            "specific_heat": specific_heat
                        }
                    except ValueError:
                        pass
                in_material = False

    # --- Fenster (SimpleGlazing) ---
    in_window_material = False
    current = []
    for line in lines:
        line = line.strip()
        if line.startswith("WindowMaterial:SimpleGlazingSystem"):
            in_window_material = True
            current = []
            continue
        if in_window_material:
            current.append(remove_comment(line))
            if ";" in line:
                if len(current) >= 3:  # robust: Name, U, SHGC
                    name = current[0]
                    try:
                        u_value = float(current[1])
                        shgc = float(current[2])
                        materials[name] = {
                            "u_factor": u_value,
                            "shgc": shgc
                        }
                    except ValueError:
                        pass
                in_window_material = False

    # --- Construction-Blöcke einsammeln ---
    construction_blocks = []
    in_construction = False
    current_block = []
    for line in lines:
        line = line.strip()
        if line.startswith("Construction,"):
            in_construction = True
            current_block = [line]
            continue
        if in_construction:
            current_block.append(line)
            if ";" in line:
                construction_blocks.append("\n".join(current_block))
                in_construction = False

    # Projekt-/Dummy-Konstruktionen rausfiltern
    filtered_blocks = []
    for block in construction_blocks:
        lines_in_block = block.splitlines()
        if len(lines_in_block) > 1 and "project" not in lines_in_block[1].lower():
            filtered_blocks.append(block)

    # Namen & Layer extrahieren
    construction_names = []
    construction_layers = []
    for block in filtered_blocks:
        lines_in_block = block.splitlines()
        if len(lines_in_block) >= 2:
            name = lines_in_block[1].split("!-")[0].strip(" ,;\n")
            layers = []
            for line in lines_in_block[2:]:
                if "!- layer" in line.lower() or "!- outside layer" in line.lower():
                    layer_name = line.split("!-")[0].strip(" ,;\n")
                    layers.append(layer_name)
                elif ";" in line:
                    layer_name = line.split("!-")[0].strip(" ,;\n")
                    layers.append(layer_name)
                    break
                else:
                    layer_name = line.strip(" ,;\n")
                    layers.append(layer_name)
            construction_names.append(name)
            construction_layers.append(layers)

    # R_total & C_total berechnen
    construction_results = []
    for name, layers in zip(construction_names, construction_layers):
        total_R = 0.0
        total_C = 0.0
        u_factor = None
        shgc = None
        is_window = False

        for layer in layers:
            layer_clean = layer.lower().strip()
            mat = materials.get(layer_clean)
            if not mat:
                continue
            if "u_factor" in mat:
                # Fensterkonstruktion
                is_window = True
                u_factor = mat["u_factor"]
                shgc = mat["shgc"]
                break
            else:
                d = mat["thickness"]
                lam = mat["conductivity"]
                rho = mat["density"]
                c = mat["specific_heat"]
                total_R += d / lam
                total_C += rho * c * d

        if is_window:
            construction_results.append((name, None, None, u_factor, shgc))
        else:
            # >>> HIER: Halbierung für BPS-InnerFloor / BPS-InnerWall <<<
            if should_halve(name):
                total_R /= 3.0
                total_C /= 2.0
            else:
                total_R /= 3.0
                total_C /= 2.0

            construction_results.append(
                (name, round(total_R, 4), round(total_C, 1), None, None)
            )
    if idf_parameter:
        df = pd.DataFrame(
            construction_results,
            columns=["Construction", "R_total [m²K/W]", "C_total [J/K·m²]", "U-Factor [W/m²K]", "SHGC"]
        )
        # df["Aussenwaermeuebergangswiderstand [m²K/W]"] = [0, 0, 0.04, 0, 0, 0, 0, 0.04] + df["R_total [m²K/W]"]
        df["Aussenwaermeuebergangswiderstand [m²K/W]"] = df["R_total [m²K/W]"]
        #df["Innenwaermeuebergangswiderstand [m²K/W]"] = 3 * [0.17, 0.13, 0.13, 0, 0.13, 0, 0, 0.10]

        #vals = np.array([0.17, 0.13, 0.13, 0, 0.13, 0, 0, 0.10]) * 3

        # DIN
        h = {"GroundFloor": 0.7, "InnerFloor": 0.7, "OuterWall": 2.5, "InnerWall": 2.5, "Roof": 5}
        # TARP, Durchschnitt
        # h = {"GroundFloor": 0.774, "InnerFloor": 0.825, "OuterWall": 1.443, "InnerWall": 1.188, "Roof": 1.899}
        # TARP, Min
        # h = {"GroundFloor": 0.350, "InnerFloor": 0.525, "OuterWall": 1.096, "InnerWall": 0.437, "Roof": 1.457}
        # TARP, Max
        # h = {"GroundFloor": 1.563, "InnerFloor": 1.235, "OuterWall": 1.586, "InnerWall": 1.356, "Roof": 2.136}

        # R_in = 1/h  (Fenster/Türen = 0, weil separat behandelt)
        vals = np.array([
            1.0 / h["GroundFloor"],  # GroundFloor
            1.0 / h["InnerFloor"],  # InnerFloor
            1.0 / h["OuterWall"],  # OuterWall
            0.0,  # Window
            1.0 / h["InnerWall"],  # InnerWall
            0.0,  # InnerDoor
            0.0,  # OuterDoor
            1.0 / h["Roof"],  # Roof
        ])

        df["Innenwaermeuebergangswiderstand [m²K/W]"] = vals
        #df["Innenwaermeuebergangswiderstand [m²K/W]"] =  [0.17, 0.13, 0.13, 0, 0.13, 0, 0, 0.10]
        return df
    else:
        df = pd.DataFrame(
            construction_results,
            columns=["Construction", "Layers", "R_total [m²K/W]", "C_total [J/K·m²]", "U-Factor [W/m²K]", "SHGC"]
        )
        df["Aussenwaermeuebergangswiderstand [m²K/W]"] = [0.04, 0.13, 0.04, 0, 0.13, 0, 0, 0.04] + df["R_total [m²K/W]"]
        df["Innenwaermeuebergangswiderstand [m²K/W]"] =  [0.17, 0.13, 0.13, 0, 0.13, 0, 0, 0.10]
        return df

def get_idf_data():
    idf_path = RESULTS_FOLDER.joinpath(
        "SFH_MPCRom_monovalent_spawn/00_DymolaWorkDir/~FMUOutput/resources/HOM_CISBAT_CondFD_V960.idf")
    try:
        with open(idf_path, "r") as file:
            lines = file.readlines()
    except FileNotFoundError as e:
        print(e)
        print(" Datei nicht gefunden, BESGRICONOP wird verwendet")
        idf_path = "D:/fwu-ssr/besgriconop/Modelica/BESGriConOp/Resources/BuildingModels/HOM_CISBAT_CondFD_V960.idf"
        with open(idf_path, "r") as file:
            lines = file.readlines()
    except:
        print("BESGRICONOP fehlerhaft, irgendwas stimmt hier nicht")

    materials_df = get_material_and_construction_data(lines)
    surfaces = parse_surface_blocks(lines)
    roof_df = extract_surface_data(surfaces, ["roof"])
    decken_df = extract_surface_data(surfaces, ["ceiling", "decke"])
    # --> Differenziere Bodentypen
    boden_df = extract_surface_data(surfaces, ["GroundFloor", "InnerFloor"])
    waende_df = extract_surface_data(surfaces, ["wall"])
    innerwall_df = waende_df[waende_df["construction"].str.contains("InnerWall", case=False, na=False)]
    outerwall_df = waende_df[waende_df["construction"].str.contains("OuterWall", case=False, na=False)]
    Roof_table = create_multicolumn_table(roof_df, "Dach")
    decke_table = create_multicolumn_table(decken_df, "Decke")
    groundfloor_table = create_multicolumn_table(boden_df[boden_df["surface_type"] == "GroundFloor"], "GroundFloor")
    innerfloor_table = create_multicolumn_table(boden_df[boden_df["surface_type"] == "InnerFloor"], "InnerFloor")
    innerwall_table = create_multicolumn_table(innerwall_df, "InnerWall")
    outerwall_table = create_multicolumn_table(outerwall_df, "OuterWall")
    flaechen_result = Roof_table
    flaechen_result = flaechen_result.join(groundfloor_table, how="outer")
    flaechen_result = flaechen_result.join(innerfloor_table, how="outer")
    flaechen_result = flaechen_result.join(decke_table, how="outer")
    flaechen_result = flaechen_result.join(innerwall_table, how="outer")
    flaechen_result = flaechen_result.join(outerwall_table, how="outer").fillna(0)
    # Optional: Fensterzuordnung, falls weiter benutzt
    wall_map = get_wall_surface_mapping(lines)
    window_df = get_exact_window_data(lines)
    window_result = window_df.merge(wall_map, how="left", left_on="building_surface", right_index=True).reset_index(drop=True)
    zone_name_map = {
        "JIlBdoXH9kyILsDvu4wx8A": "bedroom",
        "TKJwNLssk0WyOEqTNi9V3g": "livingroom",
        "ewAwLYZUK0GDV9RQVVrdsw": "kitchen",
        "Jo5bB3uKtUesyY40h7buXA": "hobby",
        "8VvlyRmVH0C1HUd3CNvpWg": "wcstorage",
        "VxcvwqdxJ0CrsbXjgBdn2A": "corridor",
        "BGVPLnC4OUeqffvCvT6TTQ": "children",
        "amiO3KhG402UMEU7Fs9xaA": "corridor2",
        "E4XJpmy03kW3qfdfjVrPLA": "bath",
        "uFb0fbIbnUCa0AXLT56UfA": "children2",
        "IfKJfrdT40ehbMFAOP2OHQ": "attic"
    }
    flaechen_result.index = flaechen_result.index.map(lambda z: zone_name_map.get(z, z))
    window_result["zone"] = window_result["zone"].map(lambda z: zone_name_map.get(z, z))

    flaechen_result["Volume [m3]"] = [33.6336014888741, 33.63362412720206, 33.63359905891071, 199.99009594462595, 59.97998511605625, 33.63362412720313, 59.979956550043006, 39.90168619860972, 39.9016910585309, 48.666794854189206, 48.66683259375543]

    return materials_df, flaechen_result, window_result

if __name__ == "__main__":
    material: pd.DataFrame
    zone_construction: pd.DataFrame
    windows: pd.DataFrame

    material, zone_construction, windows = get_idf_data()
    print(material.to_string())
    print(zone_construction.to_string())
    print(windows.to_string())
