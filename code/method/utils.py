import os

def resolve_existing_path(*paths, make=False):
    for path in paths:
        if os.path.exists(path):
            return path

    if make:
        best_path = None
        best_len = -1
        for path in paths:
            parent = os.path.dirname(os.path.normpath(path))
            while parent and not os.path.exists(parent):
                new_parent = os.path.dirname(parent)
                if new_parent == parent:
                    break
                parent = new_parent
            if os.path.exists(parent):
                # Longer existing-ancestor path = more specific match =
                # stronger signal this candidate belongs to THIS machine.
                # A generic ancestor like a bare user directory should never
                # beat a real, specific ancestor like ".../GD+CITL/experiments"
                # (long) just because it happened to be checked first.
                specificity = len(os.path.normpath(parent))
                if specificity > best_len:
                    best_len = specificity
                    best_path = path
        if best_path is not None:
            os.makedirs(best_path, exist_ok=True)
            return best_path
        raise FileNotFoundError(
            f"Could not determine which path belongs to this machine: {paths}"
        )

    raise FileNotFoundError("None of the provided paths exist.")

def list_image_files(image_dir, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    files = []
    for fn in sorted(os.listdir(image_dir)):
        if fn.lower().endswith(exts):
            files.append(os.path.join(image_dir, fn))
    if len(files) == 0:
        raise ValueError(f"No image files found in: {image_dir}")
    return files

def list_folders(dir):
    folders = []
    for fn in sorted(os.listdir(dir)):
        if os.path.isdir(os.path.join(dir,fn)):
            folders.append(os.path.join(dir, fn))
    if len(folders) == 0:
        raise ValueError(f"No model folders found in: {dir}")
    return folders

def extract_timestamp_from_folder(folder_name: str) -> str:
    """
    Assumes folder names end with something like YYYYMMDD_HHMMSS.
    Returns sortable timestamp string or empty string if not found.
    """
    parts = folder_name.split("_")
    if len(parts) >= 2:
        tail = "_".join(parts[-2:])
        if len(tail) == 15 and tail[8] == "_":
            return tail
    return ""

def build_image_to_model_map(model_folders, image_files):
    """
    Match each image stem to the most recent model folder that starts with '{image_stem}_'.
    """
    image_stems = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
    image_to_model = {}

    for stem in image_stems:
        matches = [folder for folder in model_folders if os.path.basename(folder).startswith(stem + "__")]

        if len(matches) == 0:
            print(f"[WARNING] No model folder found for image: {stem}")
            continue

        # choose most recent by timestamp suffix if possible, otherwise lexicographically last
        matches_sorted = sorted(
            matches,
            key=lambda x: extract_timestamp_from_folder(os.path.basename(x))
        )
        image_to_model[stem] = matches_sorted[-1]

    return image_to_model