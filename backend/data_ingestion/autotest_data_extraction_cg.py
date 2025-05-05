"""
Created by Eli Rosales, 4/28/2025.

This script extracts the autotest service from Codegrade and exports
    - stdin.txt or stdin.json(if no text file is found)
    - stdout.txt or stdout.json(if no text file is found)
    - Execution command
    - Execution arguments
"""

# Neccessary Imports:
# codegrade: login and api interfacing
import codegrade

# API_Data: needed for extracting autotest from api
from extract_student_data_from_API import API_Data

# dotenv: import the dotenv vars for codegrade login
from dotenv import load_dotenv

# os: for loading the dotenv variables and making file directories
import os

# sys: error outputs
import sys


def mkdir(dir_path):
    """Create a directory if it does not already exist."""
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return False
    return True


def find_filepath(folder_string):
    """Over load function for adding file to a specified folder."""
    # all files will be downloaded/created in the same folder directory called "autotest files"
    folder = "autotest files"
    return os.path.join(os.getcwd(), folder, folder_string)


def extract_data(client, tests):
    """Extract the input files/fixtures uploaded for each assignment."""
    try:
        for tuple in tests:
            assi_name = tuple[0]
            foldername = os.path.join(assi_name, "test_cases")

            # Go through the fixtures and get the contents
            meta = extract_metadata(tuple)
            make_text_files(meta, foldername, client, tuple)
    except Exception as e:
        print(str(e), file=sys.stderr)


def extract_metadata(tuple):
    """Create a dictionary that holds all auto test metadata."""
    assi_name = tuple[0]
    dictionary = {}
    dictionary[assi_name] = {}
    test = tuple[1]
    dictionary[assi_name]["meta_data"] = []
    for set in test.sets:
        suites = set.suites
        for suite in suites:
            steps = suite.steps
            for step in steps:
                if not hasattr(step.data, "inputs"):
                    continue
                if not hasattr(step.data, "program"):
                    continue
                step_dict = {}
                step_dict["name"] = step.name
                step_dict[f"{step.name}_i/o"] = []
                inputs = step.data.inputs
                for input in inputs:
                    curr_io_dict = {}
                    curr_io_dict["command"] = step.data.program
                    curr_io_dict["name"] = input.name
                    curr_io_dict["args"] = input.args
                    curr_io_dict["stdin"] = input.stdin
                    curr_io_dict["output"] = input.output
                    step_dict[f"{step.name}_i/o"].append(curr_io_dict)
                dictionary[assi_name]["meta_data"].append(step_dict)
    return dictionary


def make_text_files(meta_data, folder, client, tuple):
    """Download files: TODO."""
    assi_name = tuple[0]
    test_id = tuple[1].id
    fixtures = tuple[1].fixtures
    allfixtures = extract_fixtures(client, test_id, fixtures)
    i = 1
    for category in meta_data[assi_name]["meta_data"]:
        for test in category[f"{category["name"]}_i/o"]:
            curr_folder = os.path.join(folder, str(i))
            curr_folder = find_filepath(curr_folder)
            extract_fixture_scripts(allfixtures, curr_folder)
            i += 1


def extract_fixture_scripts(fixtures, folder):
    """Download all input/output scripts from fixtures."""
    for key, value in fixtures.items():
        if "input" in key:
            new_folder = os.path.join(folder, "input_files")
        elif "output" in key:
            new_folder = os.path.join(folder, "output_files")
        else:
            continue
        if not mkdir(new_folder):
            return
        filepath = os.path.join(new_folder, key)
        with open(filepath, "w") as f:
            f.write(str(value))


def extract_fixtures(client, test_id, fixtures):
    """Extract fixtures in a dictionary."""
    autotests = {}
    for fixture in fixtures:
        fixture_id = int(fixture.id)
        curr_autotest = client.auto_test.get_fixture(
            auto_test_id=test_id, fixture_id=fixture_id
        )
        curr_autotest = curr_autotest.decode("utf-8")
        autotests[fixture.name] = curr_autotest
    return autotests


def main():
    """Login and populate the class attributes."""
    load_dotenv()
    client = codegrade.login(
        username=os.getenv("CG_USER"),
        password=os.getenv("CG_PASS"),
        tenant="University of Nevada, Las Vegas",
    )
    cg_data = API_Data(client)
    cg_data.course = cg_data.get_course(client)
    cg_data.assignments = cg_data.get_assignments()
    assignments = cg_data.assignments

    tests = []
    for assi in assignments:
        try:
            # UNREADABLE
            if assi.auto_test_id:
                test = client.auto_test.get(auto_test_id=assi.auto_test_id)
                tests.append((assi.name, test))
        except Exception as e:
            print(str(e), file=sys.stderr)

    extract_data(client, tests)


if __name__ == "__main__":
    main()
