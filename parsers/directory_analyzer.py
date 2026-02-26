import os
from typing import List, Dict, Union
from pathlib import Path

class DirectoryAnalyzer:
    def __init__(self, extensions: List[str] = None):
        self.directories = []
        self.extensions = [ext.lower() for ext in (extensions or [])]

    def add_directory(self, dir_path: Union[str, Path]):
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            self.directories.append(path)
        else:
            raise ValueError(f"Invalid directory: {dir_path}")

    def set_extensions(self, extensions: List[str]):
        self.extensions = [ext.lower() for ext in extensions]

    def analyze(self) -> Dict[str, List[str]]:
        results = {}

        for dir_path in self.directories:
            collected_files = []

            for root, _, files in os.walk(dir_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if not self.extensions or ext in self.extensions:
                        collected_files.append(os.path.join(root, file))

            results[str(dir_path)] = collected_files

        return results

# === Example usage ===
if __name__ == "__main__":
    analyzer = DirectoryAnalyzer(extensions=[".pdf"]) #, ".md", ".html", ".docx", ".txt"

    analyzer.add_directory(r"C:\Users\Niitro_musics\Desktop\Coding\AI dev_stud\from scratch\relevant document retrieval")
    # analyzer.add_directory("C:/Users/Niitro_musics/Documents")

    results = analyzer.analyze()

    for dir_path, files in results.items():
        print(f"\nScanned: {dir_path}")
        for f in files:
            print(f" - {f}")
