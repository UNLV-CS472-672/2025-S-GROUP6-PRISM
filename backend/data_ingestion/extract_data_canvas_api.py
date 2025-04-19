"""
Created by Eli Rosales, 4/7/2025.

Extract data from the canvas API and store in a JSON dictionary.
Data to extract:
  - users
  - Professor
  - assignment links
  - ...
"""

import requests

# requests: Request data from canvas API
import sys

# sys: File error
import os

# os: getenv, path.join, makedirs, getcwd (file/directory logic)
from dotenv import load_dotenv

# dotenv: environment varibales
from bs4 import BeautifulSoup

# bs4: html parsing


class Canvas_api:
    """Canvas API manager that will interact with the api and retrieve data."""

    def __init__(self):
        """Inizialize the class."""
        # Api data
        self.url = "https://canvas.instructure.com/api/v1/"
        self.course_id = "33430000000184699"
        self.COURSE_URL = (
            "https://canvas.instructure.com/api/v1/courses/33430000000184699/"
        )
        self.__HEADERS = {"Authorization": "Bearer "}
        self.id_head = "334300000"

        # Course data
        self.course = {}
        self.course_name = ""
        self.course_code = ""
        self.course_term_id = ""

        # User data
        self.__users = {}
        self.__prof = {}
        self.__assi = {}
        self.__stud = {}

        # Assignment File Data
        self.__files = {}
        return

    def set_headers(self):
        """Set the key from .env file."""
        try:
            load_dotenv()
            key = os.getenv("WC_API_KEY")
            self.__HEADERS["Authorization"] += str(key)
        except Exception as e:
            print(str(e), file=sys.stderr)

    def mkdir(self, dir_path):
        """Create a directory if it does not already exist."""
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            print(str(e), file=sys.stderr)
            return False
        return True

    def set_course(self):
        """Set the attribute .course to the json course for easy access."""
        try:
            url = self.COURSE_URL
            response = requests.get(url, headers=self.__HEADERS)
            self.course = response.json()
            if response.status_code == 404:
                raise Exception("response failed: status == 404 - not found")
        except Exception as e:
            print(str(e), file=sys.stderr)

    def get_course(self):
        """Print the course json."""
        print(self.course)

    def set_course_data(self):
        """Set the course_name, course_code, course_term_id class attributes."""
        try:
            self.course_name = self.course["name"]
            self.course_code = self.course["course_code"]
            self.course_term_id = self.course["enrollment_term_id"]
        except Exception as e:
            print(str(e), file=sys.stderr)

    def get_course_data(self):
        """Print course data."""
        print(
            "Name:\t\t",
            self.course_name,
            "\nCourse Code:\t",
            self.course_code,
            "\nTerm:\t\t",
            self.course_term_id,
        )

    # using url = https://canvas.instructure.com/api/v1/courses/33430000000184699/users/...
    def set_prof(self):
        """Set professors variable from the professors from the development course."""
        try:
            PARAMS = {
                "enrollment_type[]": "teacher",
                "include[]": "email",
                "sort": "sortable_name",
                # "order": "asc",
            }
            url = self.COURSE_URL + "/users"
            response = requests.get(url, headers=self.__HEADERS, params=PARAMS)
            self.__prof = response.json()
        except Exception as e:
            print(str(e), file=sys.stderr)
            return None

    def get_prof(self):
        """Print professor json."""
        print(self.__prof)

    def set_stud(self):
        """Set students vairable from the students in the development course."""
        try:
            PARAMS = {
                "enrollment_type[]": "student",
                "sort": "sortable_name",
            }
            url = self.COURSE_URL + "/users"
            response = requests.get(url, headers=self.__HEADERS, params=PARAMS)
            self.__stud = response.json()
        except Exception as e:
            print(str(e), file=sys.stderr)

    def get_stud(self):
        """Print student json."""
        print(self.__stud)

    def set_users(self):
        """Set the users json as a class attribute."""
        try:
            PARAMS = {
                "per_page": 150,
            }
            # 33430000000171032
            url = self.COURSE_URL + "/users"
            response = requests.get(url, headers=self.__HEADERS, params=PARAMS)
            self.__users = response.json()
            for user in self.__users:
                ace = user["email"].split("@")[0]
                user["ace_id"] = ace
        except Exception as e:
            print(str(e), file=sys.stderr)

    def get_users(self):
        """Print users json."""
        print(self.__users)

    def set_assi(self):
        """Set the assignments."""
        try:
            PARAMS = {
                # "include[]": "all_dates",
                "ordered_by": "name",
                "bucket": "past",
            }
            url = self.COURSE_URL + "/assignments"
            response = requests.get(url, headers=self.__HEADERS, params=PARAMS)
            self.__assi = response.json()
            return response.json()
        except Exception as e:
            print(str(e), file=sys.stderr)

    def get_assi(self):
        """Print all assignments."""
        print(self.__assi)

    def set_files(self):
        """Set the __files class attribute."""
        try:
            PARAMS = {"per_page": 150}
            url = self.COURSE_URL + "/files"
            response = requests.get(url, headers=self.__HEADERS, params=PARAMS)
            self.__files = response.json()
        except Exception as e:
            print(str(e), file=sys.stderr)

    def get_files(self):
        """Print all assignments."""
        print(self.__files)

    def extract_file_ids(self, description_html):
        """Extract the ids from the description."""
        try:
            soup = BeautifulSoup(description_html, "html.parser")
            ids = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if "/files/" in href:
                    id = href.split("/")[-1].split("?")[0].split("~")[1]
                    # If link is stored in the files section of the canvas course page
                    id = self.id_head + id
                    ids.append(id)
            return ids
        except Exception as e:
            print(str(e), file=sys.stderr)

    def get_file_via_id(self, id):
        """Retrieve the file object from api using id we get from the description(of assignment)."""
        try:
            PARAMS = {"per_page": 150}
            url = self.COURSE_URL + "/files/" + id
            response = requests.get(url, headers=self.__HEADERS, params=PARAMS)
            return response.json()
        except Exception as e:
            print(str(e), file=sys.stderr)

    def download_files(self, ids, path):
        """Download the files from the extracted link urls."""
        try:
            for id in ids:
                file = self.get_file_via_id(id)
                name = file["filename"]
                url = file["url"]
                filepath = os.path.join(path, name)

                # Make the request with your Canvas token
                response = requests.get(url, headers=self.__HEADERS)

                if not self.mkdir(path):
                    return

                if response.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    print(f"Downloaded file with file ID: {name}")
                else:
                    print(f"Failed to download: {url} (Status {response.status_code})")
        except Exception as e:
            print(str(e), file=sys.stderr)


def main():
    """Set the object and call the references for testing."""
    canvas_data = Canvas_api()
    canvas_data.set_headers()
    canvas_data.set_prof()
    canvas_data.set_stud()
    canvas_data.set_users()
    assis = canvas_data.set_assi()
    canvas_data.set_course()
    canvas_data.set_course_data()
    canvas_data.set_files()
    # Print all the attributes we set from the code above.
    # canvas_data.get_prof()
    # canvas_data.get_stud()
    # canvas_data.get_users()
    # canvas_data.get_assi()
    # canvas_data.get_course()
    # canvas_data.get_course_data()
    for assi in assis:
        if not assi.get("attachment"):
            filepath = os.path.join(
                os.getcwd(), "canvas_attachments", f"{assi["name"]}"
            )
            # If there are no attachment attributes in given assignment
            # then that means the links are in the description.
            ids = canvas_data.extract_file_ids(assi["description"])
            canvas_data.download_files(ids, filepath)


if __name__ == "__main__":
    main()
