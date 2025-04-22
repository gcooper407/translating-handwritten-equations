import xml.etree.ElementTree as ET
from pathlib import Path

#Creates a txt file with the ground truth inkml LaTeX equations
#Will be able to compare later since its the ground truth
def parse_and_cache(inkml_root, out_root, split, ns={"inkml": "http://www.w3.org/2003/InkML"}):
    input_folder = Path(inkml_root) / split
    output_txt_folder = Path(out_root) / "TXT" / split
    output_txt_folder.mkdir(parents=True, exist_ok=True)

    for inkml_file in input_folder.glob("**/*.inkml"):
        out_txt = output_txt_folder / f"{inkml_file.stem}.txt"
        if out_txt.exists():
            continue

        tree = ET.parse(inkml_file)
        root = tree.getroot()
        annotation = root.find('.//inkml:annotation[@type="truth"]', ns)

        if annotation is not None and annotation.text:
            with open(out_txt, "w") as f:
                f.write(annotation.text.strip(" $"))

parse_and_cache(
    "./TC11_CROHME23/INKML/train",
    "./CACHED_CROHME",
    "CROHME2023_train"
)
