import os
import pandas as pd


def relpath(*relative_path):
    return os.path.join(os.path.dirname(__file__), *relative_path)


def input_paths(paths: dict):
    paths_ob = {}
    for k, v in paths.items():
        paths_ob[k] = relpath(v)
    return paths_ob


class Utils:
    def __init__(self, instance_name, allow_instance_rewrite, warnings):
        self.input_paths = input_paths
        self.instance_name = instance_name
        self.allow_instance_rewrite = allow_instance_rewrite
        self.warnings = warnings

    def check_file_paths(self):
        """
        If allow instance rewrite is True, all files in the output folder will be overwritten.
        """

        if self.allow_instance_rewrite and self.warnings is True:
            proceed = input("Instance rewrite enabled. Continue? (y/n): ")
            if proceed == "n":
                exit()

        if self.allow_instance_rewrite is False and os.path.exists(
            self.relpath(self.instance_name)
        ):
            print(f"Instance rewrite disabled.")
            print(
                f"The output file {self.instance_name} already exists. Delete it or use a different instance name"
            )
            exit()

    def relpath(self, *relative_path):
        return os.path.join(os.path.dirname(__file__), *relative_path)

    def to_csv(self, df: pd.DataFrame, relative_path: str):
        output_path = self.relpath(self.instance_name, relative_path)

        # Create the directory structure if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(f"{output_path}", index=False)
