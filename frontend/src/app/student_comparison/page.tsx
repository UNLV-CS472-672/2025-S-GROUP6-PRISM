"use client";
import Image from "next/image";
import styles from "./page.module.css";
import { useState, useEffect, useCallback } from "react";
import { FormControl, InputLabel, MenuItem, Select } from "@mui/material";
import Dropdown from "../components/Dropdown";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Paper from "@mui/material/Paper";
import Button from "@mui/material/Button";
import { easyFetch } from "@/utils/fetchWrapper";
import { RepeatOneSharp } from "@mui/icons-material";
import TextField from "@mui/material/TextField";
import Autocomplete from "@mui/material/Autocomplete";

//todo: Just implement the text fields and button on page
export default function StudentComparison() {
  //These are variables needed for work and their default values
  const [potentialClasses, setPotentialClasses] = useState<any[]>([]);
  const [possibleAssignments, setPossibleAssignments] = useState<any[]>([]);
  const [selectedStudentA, setSelectedStudentA] = useState(-1);
  const [selectedStudentB, setSelectedStudentB] = useState(-1);
  const [selectedClass, setSelectedClass] = useState(-1);
  const [studentABool, setStudentABool] = useState(false);
  const [studentBBool, setStudentBBool] = useState(false);
  const [potentialStudents, setPotentialStudents] = useState<any[]>([]);
  const [renderTable, setRenderTable] = useState(false);
  const [enableAutocomplete, setEnableAutocomplete] = useState(true);

  //needs input and correct function
  const onClassSelect = () => {
    //input here
    fetchStudents();
    setEnableAutocomplete(false);
  };

  const onStudentASelect = (item: number | -1) => {
    if (item == -1) {
      setStudentABool(false);
    } else {
      setStudentABool(true);
      setSelectedStudentA(item);
    }
  };

  const onStudentBSelect = (item: number | -1) => {
    if (item == -1) {
      setStudentBBool(false);
    } else {
      setStudentBBool(true);
      setSelectedStudentB(item);
    }
  };

  const handleButtonClick = () => {
    if (studentABool && studentBBool) {
      fetchAssignments(selectedStudentA, selectedStudentB);
      setRenderTable(true);
    } else {
      setRenderTable(false);
    }
  };

  const fetchClasses = useCallback(async () => {
    //const params = new URLSearchParams({
    //  professor_id: { pid },
    //});

    const response = await fetch(
      `http://localhost:8000/api/course/details${params}`,
      {
        method: "get",
      }
    );

    const data = await response.json();

    if (response.ok) {
      console.log("Fetched classes:", data);
      setPotentialClasses(data);
    }
  }, []);

  const fetchStudents = useCallback(async (classInstanceId: number) => {
    const response = await fetch(
      `http://localhost:8000/api/course/details?course_instance=${classInstanceId}`,
      {
        method: "get",
      }
    );

    const data = await response.json();

    if (response.status === 200) {
      console.log("GOOD REQUEST");
      setPotentialStudents(data.students.results);
    }
  }, []);

  const fetchAssignments = useCallback(
    async (student_one_id: number, student_two_id: number) => {
      //const params = new URLSearchParams({
      //  course_instance: { cid },
      //});
      const response = await fetch(
        `http://localhost:8000/api/assignment/compare?student_one=${student_one_id}&student_two=${student_two_id}`,
        {
          method: "get",
        }
      );

      const data = await response.json();

      if (response.status === 200) {
        console.log("GOOD REQUEST");
        setPossibleAssignments(data);
      }
    },
    []
  );

  return (
    //The TextField accepts and returns a string written by the user, The dropdown returns an id correlating to a specific
    // class, and the button will handle submitting the data to the database
    <Stack spacing={1} sx={{ justifyContent: "center", alignItems: "center" }}>
      {
        //needs onClassSelect function added
      }
      <FormControl fullWidth>
        <InputLabel id="demo-simple-select-label">Classes</InputLabel>
        <Select
          labelId="demo-simple-select-label"
          id="demo-simple-select"
          label="Age"
          onOpen={fetchClasses}
        >
          {potentialClasses?.map((value, index) => (
            <MenuItem key={index} value={value}>
              {value["name"]}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      <Stack direction="row">
        <Autocomplete
          disablePortal
          disabled={!enableAutocomplete}
          //input data stuff below
          options={potentialStudents}
          getOptionLabel={(option) =>
            `${option.first_name} ${option.last_name}`
          }
          sx={{ width: 300 }}
          onChange={(a, b) => {
            if (b != null) {
              onStudentASelect(b!.id);
            } else {
              onStudentASelect(-1);
            }
          }}
          renderInput={(params) => (
            <TextField {...params} label={"Student A"} color="primary" />
          )}
        />

        <Autocomplete
          disablePortal
          disabled={!enableAutocomplete}
          //input data stuff below
          options={potentialStudents}
          getOptionLabel={(option) =>
            `${option.first_name} ${option.last_name}`
          }
          sx={{ width: 300 }}
          onChange={(a, b) => {
            if (b != null) {
              onStudentBSelect(b!.id);
            } else {
              onStudentBSelect(-1);
            }
          }}
          renderInput={(params) => (
            <TextField {...params} label={"Student B"} color="primary" />
          )}
        />
      </Stack>
      {
        <Button
          variant="outlined"
          onClick={() => {
            handleButtonClick();
          }}
        >
          Select Students to compare
        </Button>
      }
      <Typography variant="h5" gutterBottom>
        {"Overall - " + 0}
      </Typography>
      {renderTable && (
        <TableContainer component={Paper}>
          <Table
            sx={{ width: 650, justifyContent: "center" }}
            aria-label="simple table"
          >
            <TableHead>
              <TableRow>
                <TableCell>Student Name</TableCell>
                <TableCell>Relative</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {possibleAssignments.map((row) => (
                <TableRow
                  key={row.title}
                  sx={{ "&:last-child td, &:last-child th": { border: 0 } }}
                >
                  <TableCell component="th" scope="row">
                    {row.title}
                  </TableCell>
                  <TableCell>
                    {row.similarity_pair.results.percentage}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Stack>
  );
}
