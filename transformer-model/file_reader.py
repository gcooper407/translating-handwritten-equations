from pathlib import Path

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def parse_and_cache_inkml(inkml_dir, cache_dir, loc, ns={"inkml": "http://www.w3.org/2003/InkML"}):
  """
  Parses InkML files, extracts stroke data and LaTeX strings, and caches them as PNG and TXT files.

  Args:
    inkml_dir (str): Path to the directory containing InkML files.
    cache_dir (str): Path to the directory where cached files will be stored.
    ns (dict): Namespace for parsing InkML files.
  """
  inkml_dir = Path(inkml_dir) / f"{loc}/"

  img_cache_dir = Path(cache_dir) / f"IMG/{loc}/"
  img_cache_dir.mkdir(parents=True, exist_ok=True)

  txt_cache_dir = Path(cache_dir) / f"TXT/{loc}/"
  txt_cache_dir.mkdir(parents=True, exist_ok=True)

  curr = 0

  for inkml_file in inkml_dir.glob("**/*.inkml"):
    
    # Define cache file paths
    img_file = img_cache_dir / f"{inkml_file.stem}.png"
    txt_file = txt_cache_dir / f"{inkml_file.stem}.txt"

    # Skip if already cached
    if img_file.exists() and txt_file.exists():
      continue

    # Parse InkML file
    tree = ET.parse(inkml_file)
    root = tree.getroot()

    # Extract strokes
    strokes = []
    for trace in root.findall(".//inkml:trace", ns):
      coords = trace.text.strip().split(",")
      coords = [
        (float(x), -float(y))  # Invert y-axis to match InkML's coordinate system
        for x, y, *z in [coord.split() for coord in coords]
      ]
      strokes.append(coords)

    # Extract LaTeX string
    latex_string = root.find('.//inkml:annotation[@type="truth"]', ns)

    # Cache LaTeX string in a text file
    with open(txt_file, "w") as f:
      f.write(latex_string.text.strip(" $"))

    # Render strokes and save as PNG
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_aspect("equal")
    for coords in strokes:
      x, y = zip(*coords)
      ax.plot(x, y, color="black", linewidth=2)
    plt.savefig(img_file, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    curr += 1
    print(curr)


# Example usage
inkml_directory = "./TC11_CROHME23/INKML/"
cache_directory = "./CACHED_CROHME/"

# Print the total number of files
# print("Total files found:", len(list(Path(inkml_directory).glob("**/*.inkml"))))

parse_and_cache_inkml(inkml_directory, cache_directory, "train")
# parse_and_cache_inkml(inkml_directory, cache_directory, "val")