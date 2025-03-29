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
import NextLink from "next/link";

// Example data
const classSections = [
  { className: "Math 101", sections: ["Section A", "Section B"] },
  { className: "Physics 202", sections: ["Section C"] },
];

const staticLinks = [
  { label: "Account", href: "/account" },
  { label: "Alerts", href: "/alerts" },
  { label: "Dashboard", href: "/dashboard" },
  { label: "Student Comparison", href: "/student_comparison" },
];

// A component to wrap a Link component in Next.js to forward refs to an <a> element
// HTMLAnchorElement tells TypeScript that the ref is for an <a>
const LinkListItem = forwardRef<HTMLAnchorElement, { href: string; children: React.ReactNode }>(
    ({ href, children, ...props }, ref) => (
        // passHref passes href to <a> and legacyBehavior is used to manually write the <a> element
      <Link href={href} passHref legacyBehavior>
        {/* ref={ref} forwards things */}
        <a ref={ref} {...props}>
          {children}
        </a>
      </Link>
    )
  );
// Set friendly name for the component
LinkListItem.displayName = "LinkListItem";

const HeaderBar = ({ title }: { title: string }) => {
  const [open, setOpen] = useState(false);

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
                <ListItemButton key={link.href} component={LinkListItem} href={link.href}>
                    <ListItemText primary={link.label} />
                </ListItemButton>
            ))}
          </List>
        </Box>
      </Drawer>
    </>
  );
};

export default HeaderBar;
