import React, { useState, forwardRef } from "react";
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Divider,
  Box,
  ListSubheader,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import HomeIcon from "@mui/icons-material/Home";
import Link from "next/link";
import { SignOutButton } from "@/components/AuthenticationMethod";
import { useEffect } from "react";
import { easyFetch } from "@/utils/fetchWrapper";
import NextLink from "next/link";

// Example data
const classSections = [
  { className: "Math 101", sections: ["Section A", "Section B"] },
  { className: "Physics 202", sections: ["Section C"] },
];

// Real data
interface ClassInstance {
  id: number;
  name: string;
}

interface Section {
  id: number;
  section_number: number;
  class_instance: {
    id: number;
    name: string;
  };
  semester: {
    name: string;
  }
}

const staticLinks = [
  { label: "Account", href: "/account" },
  { label: "Alerts", href: "/alerts" },
  { label: "Dashboard", href: "/dashboard" },
  { label: "Student Comparison", href: "/student_comparison" },
];

const HeaderBar = ({ title }: { title: string }) => {
  const [open, setOpen] = useState(false);
  const [courses, setCourses] = useState<any[]>([])

  useEffect(() => {
    const fetchCourses = async () => {
      try {
        const response = await easyFetch(
          "http://localhost:8000/api/course/courseinstances/",
          {
            method: "get",
          }
        )
  
        const data = await response.json()
  
        if (response.ok) {
          setCourses(data["results"])
        }
      } catch (error) {
        console.error(error)
      }
    }
      fetchCourses()
  }, [])

  const toggleDrawer = (state: boolean) => () => {
    setOpen(state);
  };

  return (
    <>
      <AppBar position="static" color="default" elevation={1}>
        <Toolbar>
          <IconButton edge="start" color="inherit" onClick={toggleDrawer(true)} aria-label="menu">
            <MenuIcon />
          </IconButton>

          <Link href="/" passHref>
            <IconButton edge="start" color="inherit" aria-label="home" sx={{ ml: 1 }}>
              <HomeIcon />
            </IconButton>
          </Link>

          <Typography variant="h6" component="div" sx={{ flexGrow: 1, textAlign: "center" }}>
            {title}
          </Typography>
        </Toolbar>
      </AppBar>

      <Drawer anchor="left" open={open} onClose={toggleDrawer(false)}>
        <Box sx={{ width: 250 }} role="presentation" onClick={toggleDrawer(false)}>
          <List
            subheader={<ListSubheader component="div">Classes</ListSubheader>}
          >
            {classSections.map((cls) => (
              <Box key={cls.className} sx={{ pl: 2 }}>
                <Typography variant="subtitle1">{cls.className}</Typography>
                {cls.sections.map((sec) => (
                  <ListItem key={sec} dense>
                    <ListItemText primary={sec} />
                  </ListItem>
                ))}
              </Box>
            ))}
          </List>

          <Divider />

          <List
            subheader={<ListSubheader component="div">Navigation</ListSubheader>}
          >
            {staticLinks.map((link) => (
                <ListItemButton key={link.href} component={Link as React.ElementType} href={link.href}>
                    <ListItemText primary={link.label} />
                </ListItemButton>
            ))}
          </List>
          <SignOutButton />
        </Box>
      </Drawer>
    </>
  );
};

export default HeaderBar;
