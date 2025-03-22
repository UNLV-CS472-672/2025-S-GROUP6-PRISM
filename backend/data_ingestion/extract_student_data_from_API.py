'''
    Created by Eli Rosales, 3/2/2025

    This script is a part of the Data Ingestion phase of PRISM.
    This script will automate the need for manual data extraction from 
    codegrade.
    It will extract the csv with 
'''
import io
import os
import sys
import time
import zipfile
import codegrade
import httpx
import shutil
import json
import csv
from codegrade.utils import select_from_list
from dotenv import load_dotenv

class API_Data:
    #variables
    '''
    1. client                           (cg client)                                                                 #CODEGRADE AUTHENTICATED CLIENT
    2. course                           (cg courseService)                                                          #STR
    3. course_name                      (str)                                                                       #STR
    4. all_assignments []               ([(assignment_id,assignemnt_name),(assignment_id,assignemnt_name)...])      #LIST OF TUPLES
    5. rubric {}                        ({"assignment_id":  {header1:[header_name,points],header2:'',...}})         #DIC
    6. all_assignment_submissions {}    ({"assignment_id":  [sub_id,sub_id...] })                                   #DIC
    7. graders {}                       ([[grader_name,grader_username,grader_user_id],...])                        #LIST OF LISTS
    8. rubric_grades {}                 ([("sub_id - user.name",points achieved)]})                                 #LIST OF TUPLES
    9. course_info                      ({'id':id,  'name':name,    'date':date})
    
    10. submisions
    11. assignments
    12. create folder path              path to the 'CREATED_FILES' directory(storing all files made by this script)
    '''
    def __init__(self,client):
        #main will handle the username and password part(retrieve from frontend)
        #self.__username                     = username (parameter)  #__private_var
        #self.__password                     = password (parameter) #__prvate_var
        self.client                         = client
        self.course_name                    = ""
        self.course                         = self.get_course(client)
        self.assignments                    = self.get_assignments()
        self.create_folder_path             = ""
        #To output in a human readable format
        self.course_info                    = {}
        self.all_assignments = self.graders = self.rubric_grades = []
        self.rubrics                        = {} #{"assignment_id": {}, "assignment_id": {}}
        self.all_assignment_submissions     = {} #{"assignment_id": [], "assignment_id": []}
        
#helper functions
    def handle_maybe(self,maybe):
        return maybe.try_extract(lambda: SystemExit(1))

    def mkdir(self,dir):
        try:
            os.makedirs(dir, exist_ok=True)
        except Exception as e:
            print(str(e), file=sys.stderr)
            return False
        else:
            return True

    def get_course(self,client):
        try:
            course = self.handle_maybe(select_from_list(
                    'Select a course',
                    client.course.get_all(),
                    lambda c: c.name,
                    ))
        except Exception as e:
            return e
        else:
            self.course_name = course.name
            #self.course_info = self.get_course_info(course)
            return course
#Populate class vars and output data in a human readable format
    def get_assignments(self):
        return self.course.assignments
    #TODO vvv finish these methods
    '''
    def get_course_info(self):
        return None
    def get_all_assignments(self):
        return None
    def get_all_graders(self):
        return None
    def get_rubric_grades(self):
        return None
    def get_rubrics(self):
        return None
    def get_all_assignment_subs(self):
        return None
    '''

#Getters from API:
    #ASSIGNMENT SERVICE TYPE
    def get_all_graders(self, assignment):
        try:
            graders = self.client.assignment.get_all_graders(assignment_id=assignment.id)
        except Exception as e:
            print(str(e))
        else:
            return graders
    
    def get_rubric(self, assignment):
        try:
            rubric = self.client.assignment.get_rubric(assignment_id=assignment.id)
        except Exception as e:
            print(str(e))
        else:
            return rubric

    def get_desc(self,assignment):
        try:
            desc = self.client.assignment.get_description(assignment_id=assignment.id)
        except Exception as e:
            print(str(e))
        else:
            return desc
        
    def get_time_frames(self,assignment):
        try:
            times = self.client.assignment.get_timeframes(assignment_id=assignment.id)
        except Exception as e:
            print(str(e))
        else:
            return times

    def get_feedback(self,assignment):
        try:
            feedback = self.client.assignment.get_all_feedback(assignment_id=assignment.id)
        except Exception as e:
            print(str(e))
        else:
            return feedback
        
    '''UNUSED FUNCTIONS maybe we can use them
    def get_cg_auto_test(client,assignment):
        try:
            test = client.assignment.get_auto_test(assignment_id=assignment.id)
        except Exception as e:
            print(str(e))
        else:
            return test
    '''
    #COURSE SERVICE TYPES:
    def get_users(self,course):    
        try:
            users = self.client.course.get_all_users(course_id=course.id)
        except Exception as e:
            print(str(e))
        else:
            return users
        
    def get_all_user_submissions(self,course,user_id):
        #{'assignement_id':[extendedWork(...)]}
        try:
            users = self.client.course.get_submissions_by_user(course_id=course.id,user_id=user_id)
        except Exception as e:
            print(str(e))
        else:
            return users

    #SUBMISSION SERVICE TYPES:
    def get_rubric_grade(self,submission_id):
        try:
            grade = self.client.submission.get_rubric_result(submission_id=submission_id)
        except Exception as e:
            return None
        else:
            return grade



#EXTRACTION
    def get_json_file(self,stud_dict,file_path):
        #file_name
        file_output = os.path.join(file_path,'.cg-info.json')
        print('Making json file ".cg-info.json"')
        #check if path exists
        if not self.mkdir(file_path):
            return
        with open(file_output,'w') as file:
            json.dump(stud_dict,file)

    def download_submission(self, submission, output_dir, *, retries=5):
        try:
            zipinfo = self.client.submission.get(
                submission_id=submission.id,
                type='zip',
            )
            # zipdata = client.file.download(filename=zipinfo['name'])
            zipdata = self.client.file.download(filename=zipinfo.name)

        except httpx.ReadError:
            # This is raised sometimes when doing a lot of simultaneous
            # downloads in a short amount of time.
            if not retries:
                raise
            time.sleep(1)
            return self.download_submission(
                self.client,
                submission,
                output_dir,
                retries=retries - 1,
            )

        if submission.user.group:
            username = submission.user.group.name + ' (Group)'
        else:
            username = submission.user.name

        student_output_dir = os.path.join(output_dir, username)
        #C:\path\...\CS 472 - Development - Businge - Assignment 3\Jacob Kasbohm

        if not self.mkdir(student_output_dir):
            return

        with zipfile.ZipFile(io.BytesIO(zipdata), 'r') as zipf:
            try:
                for file in zipf.namelist():
                    filename = os.path.basename(file)
                    if not filename:
                        continue
                    source = zipf.open(file)
                    target = open(os.path.join(student_output_dir, filename), "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)    
            except zipfile.BadZipFile:
                print('Invalid zip file', file=sys.stderr)

    #helper function for extract_all_assignments
    def get_output_dir(self,course_name,assignment):
        self.create_folder_path = os.path.join(os.getcwd(),"CREATED_FILES")
        fileName = course_name + " - " + assignment.name
        default_dir = os.path.join(self.create_folder_path,fileName)
        return default_dir
    #helper function for extract_all_assignments
    def get_sorted_dict (self,studDict):
        studDict["submission_ids"] = sorted(studDict["submission_ids"].items())
        studDict["user_ids"] = sorted(studDict["user_ids"].items())
        for key1 in studDict:
            currDict = {}
            for key,value in studDict[key1]:
                currDict[key] = value
            studDict[key1] = currDict
        return studDict
    #helper function for extract_all_assignments
    def make_zip_archive(self,zip_file_name,dir_path):
        """
        Creates a zip archive of a directory.

        Args:
            zip_file_name (str): name of the resulting output .zip file
            dir_path (str): Path to the directory to be zipped.
        """
        if not os.path.exists(dir_path):
            print(f"Unable to create '{dir_path}.zip' due to unfound '{zip_file_name}' directory name.")
            return
        destPath = os.path.join(self.create_folder_path,zip_file_name)
        #store into the new path instead of current dir
        shutil.make_archive(destPath,'zip', dir_path)
        #base_dir=zip_loc, root_dir=zip_loc, format='zip', base_name=zip_dest
        print(f"SUCCESS!! file '{zip_file_name}.zip' has been created!")
        #os.rename(os.path.splitext(zip_path)[0] + '.zip', zip_path)

    def extract_all_assignments(self,assignments):
        '''
        INPUT PARAMS:
        assignemnts = assignments service from codegrade.
        DESC: 
        Input all assignments from a course and create a zipfile 
        '''
        print(f"Extracting all assignments from {self.course.name}:\n")
        for assignment in assignments:
            #get all submissions at current assignment
            submissions = self.client.assignment.get_all_submissions(
                assignment_id=assignment.id,
                latest_only=True,
                )
            if len(submissions) > 0:
                output_dir = self.get_output_dir(self.course.name,assignment)
                #output_dir = 'C:\\Users\\ejera\\testenv\\CS 472 - Development - Businge - Assignment 0'
                students = {"submission_ids":{},"user_ids":{}}
                print(f"\nExtracting submission source code for {assignment.name}:")
                for submission in submissions:
                    #populate the students dictionary
                    subID = submission.id
                    stud_id_key = f"{subID} - {submission.user.name}"
                    stud_uid_key = f"{subID} - {submission.user.name}"
                    #local dictionary
                    userID = submission.user.id
                    students["submission_ids"][stud_id_key] = subID
                    students["user_ids"][stud_uid_key] = userID

                    #download all submissions and store them in the output_dir
                    print('Downloading', submission.user.name)
                    self.download_submission(submission, output_dir)
                #if there are more than one student
                students = self.get_sorted_dict(students)
                self.get_json_file(students,output_dir)
                #place json in file
                fileName = self.course.name + " - " + assignment.name 
                print(f"{fileName} Successfully completed!")
                print(f'\nMaking "{fileName}" into a zip file')
                #convert directory made into a zip archive
                self.make_zip_archive(fileName,output_dir)
            else:
                print(f'No submissions for {assignment.name}')
    #helper fucntion for extract csv
    def get_grade(self,max_grade,grade_achieved):
        #assuming all the multipliers for each grade category is x1
        complement = 100 / max_grade
        return grade_achieved*complement
    def extract_csv (self,assignments):
        '''
        INPUT PARAMS:
        assignemnts = assignments service from codegrade.
        DESC: 
        Input all assignments from a course and create csv that has all information about
        the student submisison including stud-id, username, name, and grade.
        '''
        print(f"\nExtracting .CSV file(s) for student(s) from class: '{self.course.name}'")
        for assignment in assignments:
            submissions = self.client.assignment.get_all_submissions(
                assignment_id=assignment.id,
                latest_only=True,
                )
            if len(submissions) > 0:
                #always create first row
                rows = ['Id','Username','Name','Grade']
                
                file_name = self.get_output_dir(self.course.name,assignment)
                file_name = file_name+ '.csv'
                #find file_name
                fileCSV = open(file_name, 'w',newline='')
                #open csv file - dont create a csv file unless there are submissions for 
                #the assignment
                writer = csv.writer(fileCSV)
                #write the first row
                writer.writerow(rows)
                self.get_feedback(assignment)
                for submission in submissions:
                    grade = self.get_grade(assignment.max_grade,submission.grade)
                    #find the grade
                    rows = [submission.user.id,
                            submission.user.username,
                            submission.user.name,
                            grade]
                    writer.writerow(rows)
                print("SUCCESS!!! Create file 'CS 472 - Development - Businge - Assignment 4.csv' has been created!")
                #close csv file
                fileCSV.close()
            else:
                print(f"No submissions for {assignment.name}")
            

def main():
    #log in client
    #BEFORE RUNNING!!!!!!
    #EDIT THE .ENV FILE AND INPUT YOUR USERNAME AND PASSWORD FOR USERNAME AND CG_PASSWORD
    #OR USE THIS INSTEAD TO LOGIN:
    client = codegrade.login_from_cli()
    cg_data = API_Data(client)
    cg_data.extract_all_assignments(cg_data.assignments)
    cg_data.extract_csv(cg_data.assignments)
    
if __name__ == "__main__":
    main()