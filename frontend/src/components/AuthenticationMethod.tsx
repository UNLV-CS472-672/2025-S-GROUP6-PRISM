"use client";
import { Button } from "@mui/material";
import { signIn, signOut } from "next-auth/react";

export function SignInButton() {
    return (
        <Button
            variant="contained"
            onClick={() => signIn("google", { callbackUrl: "/dashboard" })}
            data-testid="google-sign-in"
        >
            Sign In with Google
        </Button>
    );
}

export function SignOutButton() {
    return (
        <Button variant="outlined" onClick={() => signOut({ callbackUrl: "/login" })}>
            Sign Out
        </Button>
    );
}
