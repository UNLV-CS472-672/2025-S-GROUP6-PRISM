"""Created by Eli Rosales, 3/2/2025.

This script automates the extraction of CodeGrade data as part of the
PRISM Data Ingestion pipeline.
"""

import io
import os
import sys
import time
import zipfile
from zipfile import BadZipFile
import json
import csv
import shutil
import datetime

from httpx import ReadError
import codegrade
from codegrade.utils import select_from_list


class API_Data:
    """Handle data extraction from the CodeGrade API."""

    def __init__(self, client):
        """Initialize the API_Data instance with CodeGrade client."""
        self.client = client
        self.course_name = ""
        self.course = None
        self.assignments = None
        self.create_folder_path = ""
        self.course_info = {}
        self.all_assignments = []
        self.graders = []
        self.rubric_grades = []
        self.rubrics = {}
        self.all_assignment_submissions = {}

    def handle_maybe(self, maybe):
        """Safely extract value from a 'maybe' result."""
        return maybe.try_extract(lambda: SystemExit(1))

    def mkdir(self, dir_path):
        """Create a directory if it does not already exist."""
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            print(str(e), file=sys.stderr)
            return False
        return True

    def get_course(self, client):
        """Prompt user to select a course using the client."""
        try:
            course = self.handle_maybe(
                select_from_list(
                    "Select a course",
                    client.course.get_all(),
                    lambda c: c.name,
                )
            )
        except Exception as e:
            return e

        self.course_name = course.name
        return course

    def get_assignments(self):
        """Return list of assignments for the selected course."""
        return self.course.assignments

    def get_course_info(self):
        """Return basic course info as a dictionary."""
        return {
            "Course-ID": self.course.id,
            "Name": self.course.name,
            "Created-Date": self.course.created_at,
        }

    def get_rubric_grades_dict(self, assignments):
        """Build a dictionary of rubric results for each submission."""
        grade_dict = {}
        for assignment in assignments:
            submissions = self.get_all_submissions(assignment)
            for submission in submissions:
                sub_key = f"{submission.id} - {submission.user.name}"
                grade_dict[sub_key] = []
                rubric = self.get_rubric_grade(submission.id)
                self.get_rubric_value(rubric, grade_dict, sub_key)
        return grade_dict

    def get_rubric_value(self, rubric, grade_dict, sub_key):
        """Extract rubric info and append to grade_dict for a submission."""
        for i, obj in enumerate(rubric.selected):
            grade_dict[sub_key].append(
                {
                    "header": rubric.rubrics[i].header,
                    "points_achieved": obj.achieved_points,
                    "points_possible": obj.points,
                    "multiplier": obj.multiplier,
                }
            )

    def get_all_submissions(self, assignment):
        """Return all submissions for a given assignment."""
        try:
            return self.client.assignment.get_all_submissions(
                assignment_id=assignment.id
            )
        except Exception as e:
            print(str(e))

    def get_all_graders(self, assignment):
        """Return all graders for a given assignment."""
        try:
            return self.client.assignment.get_all_graders(assignment_id=assignment.id)
        except Exception as e:
            print(str(e))

    def get_rubric(self, assignment):
        """Return the rubric for a given assignment."""
        try:
            return self.client.assignment.get_rubric(assignment_id=assignment.id)
        except Exception as e:
            print(str(e))

    def get_desc(self, assignment):
        """Return the description for an assignment."""
        try:
            return self.client.assignment.get_description(assignment_id=assignment.id)
        except Exception as e:
            print(str(e))

    def get_time_frames(self, assignment):
        """Return timeframes for an assignment."""
        try:
            return self.client.assignment.get_timeframes(assignment_id=assignment.id)
        except Exception as e:
            print(str(e))

    def get_feedback(self, assignment):
        """Return all feedback for a given assignment."""
        try:
            return self.client.assignment.get_all_feedback(assignment_id=assignment.id)
        except Exception as e:
            print(str(e))

    def get_users(self, course):
        """Return all users enrolled in the given course."""
        try:
            return self.client.course.get_all_users(course_id=course.id)
        except Exception as e:
            print(str(e))

    def get_all_user_submissions(self, course, user_id):
        """Return all submissions by a specific user."""
        try:
            return self.client.course.get_submissions_by_user(
                course_id=course.id, user_id=user_id
            )
        except Exception as e:
            print(str(e))

    def get_rubric_grade(self, submission_id):
        """Return rubric grade for a given submission ID."""
        try:
            return self.client.submission.get_rubric_result(submission_id=submission_id)
        except Exception:
            return None

    def get_json_file(self, student_dict, file_path):
        """Write student_dict to a .cg-info.json file in given path."""
        if not self.mkdir(file_path):
            return

        json_path = os.path.join(file_path, ".cg-info.json")
        print('Generating ".cg-info.json"')
        with open(json_path, "w") as f:
            json.dump(student_dict, f)

    def download_submission(self, submission, output_dir, retries=5):
        """Download and extract a student's submission ZIP file."""
        try:
            zipinfo = self.client.submission.get(
                submission_id=submission.id,
                type="zip",
            )
            zipdata = self.client.file.download(filename=zipinfo.name)
        except ReadError:
            if not retries:
                raise
            time.sleep(1)
            return self.download_submission(submission, output_dir, retries=retries - 1)

        username = (
            submission.user.group.name + " (Group)"
            if submission.user.group
            else submission.user.name
        )

        student_dir = os.path.join(output_dir, username)
        if not self.mkdir(student_dir):
            return

        try:
            with zipfile.ZipFile(io.BytesIO(zipdata), "r") as zipf:
                for file in zipf.namelist():
                    filename = os.path.basename(file)
                    if not filename:
                        continue
                    with zipf.open(file) as source, open(
                        os.path.join(student_dir, filename), "wb"
                    ) as target:
                        shutil.copyfileobj(source, target)
        except BadZipFile:
            print("Invalid zip file", file=sys.stderr)

    def get_output_dir(self, course_name, assignment):
        """Return output directory path for storing extracted files."""
        self.create_folder_path = os.path.join(os.getcwd(), "cg_data")
        return os.path.join(
            self.create_folder_path, f"{course_name} - {assignment.name}"
        )

    def get_sorted_dict(self, student_dict):
        """Sort student submission/user ID dictionaries."""
        student_dict["submission_ids"] = sorted(student_dict["submission_ids"].items())
        student_dict["user_ids"] = sorted(student_dict["user_ids"].items())

        for key in student_dict:
            temp = {}
            for k, v in student_dict[key]:
                temp[k] = v
            student_dict[key] = temp

        return student_dict

    def make_zip_archive(self, zip_name, dir_path):
        """Create a zip archive from the given directory."""
        if not os.path.exists(dir_path):
            print(f"Cannot create '{zip_name}.zip'; directory not found.")
            return

        dest_path = os.path.join(self.create_folder_path, zip_name)
        shutil.make_archive(dest_path, "zip", dir_path)
        print(f"SUCCESS! File '{zip_name}.zip' has been created.")

    def extract_all_assignments(self, assignments):
        """Download, extract, and archive submissions for each assignment."""
        print(f"Extracting assignments from {self.course.name}:\n")
        now = datetime.datetime.now()

        for assignment in assignments:
            submissions = self.client.assignment.get_all_submissions(
                assignment_id=assignment.id,
                latest_only=True,
            )

            lock_date = assignment.lock_date or assignment.deadline
            lock_dt = datetime.datetime(
                lock_date.year,
                lock_date.month,
                lock_date.day,
                lock_date.hour,
                lock_date.minute,
                lock_date.second,
            )

            if not submissions:
                print(f"No submissions for {assignment.name}")
                continue
            if lock_dt > now:
                print(
                    f"Lock date ({lock_dt}) not passed yet for {
                        assignment.name}"
                )
                continue

            print(f"\nExtracting submission source code for {assignment.name}")
            output_dir = self.get_output_dir(self.course.name, assignment)
            student_dict = {"submission_ids": {}, "user_ids": {}}

            for submission in submissions:
                sub_id = submission.id
                user = submission.user
                key = f"{sub_id} - {user.name}"
                student_dict["submission_ids"][key] = sub_id
                student_dict["user_ids"][key] = user.id
                print("Downloading", user.name)
                self.download_submission(submission, output_dir)

            student_dict = self.get_sorted_dict(student_dict)
            self.get_json_file(student_dict, output_dir)

            zip_name = f"{self.course.name} - {assignment.name}"
            print(f"{zip_name} successfully completed.")
            print(f"Archiving '{zip_name}' as ZIP.")
            self.make_zip_archive(zip_name, output_dir)

    def get_grade(self, max_grade, grade_achieved):
        """Return normalized grade as a percentage."""
        return (grade_achieved / max_grade) * 100 if max_grade else 0

    def extract_csv(self, assignments):
        """Generate CSV files of assignment grades and student info."""
        print(f"\nExtracting CSVs for course: '{self.course.name}'")

        for assignment in assignments:
            submissions = self.client.assignment.get_all_submissions(
                assignment_id=assignment.id,
                latest_only=True,
            )

            lock_date = assignment.lock_date or assignment.deadline
            lock_dt = datetime.datetime(
                lock_date.year,
                lock_date.month,
                lock_date.day,
                lock_date.hour,
                lock_date.minute,
                lock_date.second,
            )

            if not submissions:
                print(f"No submissions for {assignment.name}")
                continue
            if lock_dt > datetime.datetime.now():
                print(
                    f"Lock date ({lock_dt}) not passed yet for {
                        assignment.name}"
                )
                continue

            file_path = self.get_output_dir(self.course.name, assignment) + ".csv"
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Id", "Username", "Name", "Grade"])
                self.get_feedback(assignment)
                for submission in submissions:
                    grade = self.get_grade(assignment.max_grade, submission.grade)
                    writer.writerow(
                        [
                            submission.user.id,
                            submission.user.username,
                            submission.user.name,
                            round(grade, 2),
                        ]
                    )
            print(f"SUCCESS! CSV file '{file_path}' created.")

    def delete_created_folder(self):
        """Remove created folder and all its contents."""
        try:
            shutil.rmtree(self.create_folder_path)
            print(
                f"Directory '{
                    self.create_folder_path}' deleted successfully."
            )
        except FileNotFoundError:
            print(f"Directory '{self.create_folder_path}' not found.")
        except PermissionError:
            print(f"Permission denied to delete '{self.create_folder_path}'.")
        except OSError as e:
            print(f"Error deleting directory: {e}")


def main():
    """Run the CodeGrade data ingestion pipeline."""
    client = codegrade.login_from_cli()
    cg_data = API_Data(client)
    cg_data.course = cg_data.get_course(client)
    cg_data.assignments = cg_data.get_assignments()
    cg_data.extract_all_assignments(cg_data.assignments)        #download all submissions of every assignment witht the lockdate past
    cg_data.extract_csv(cg_data.assignments)                    #extrace the csv file witht he columns [ID,Username,Name,Grade]                       
    cg_data.delete_created_folder()                             #delete the created folder
    #print(cg_data.get_course_info())
    #print(cg_data.get_rubric_grades_dict(cg_data.assignments))


if __name__ == "__main__":
    main()
