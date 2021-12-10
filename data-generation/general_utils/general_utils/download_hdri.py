import argparse
import os

import requests


def download_hdri(hdri_name, output_dir="", res="1k", format="exr"):
    """Downloads an HDRI from Poly Haven.

    Args:
        hdri_name (str): short name of the HDRI. Can be found in the url of the HDRI on polyhaven.com
        output_dir (str, optional): directory where the HDRI will be saved. Defaults to "".
        res (str, optional): HDRI resolution e.g. "1k", "2k", "4k", "8k". Defaults to "1k".
        format (str, optional): HDRI format, "hdr" or "exr". Defaults to "exr".

    Returns:
        boolean: True when the download was successful.
    """
    url = f"https://dl.polyhaven.org/file/ph-assets/HDRIs/{format}/{res}/{hdri_name}_{res}.{format}"
    # example: https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/1k/studio_small_09_1k.exr

    r = requests.get(url, allow_redirects=True)

    if r.status_code != 200:
        print(f"Bad status code {r.status_code}")
        return False

    local_filename = f"{hdri_name}_{res}.{format}"
    local_path = os.path.join(output_dir, local_filename)

    with open(local_path, "wb") as f:
        f.write(r.content)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hdri_name")
    parser.add_argument("-d", "--dir", dest="output_dir", metavar="OUTPUT_DIRECTORY", default="")
    args = parser.parse_args()

    download_hdri(args.hdri_name, args.output_dir)
