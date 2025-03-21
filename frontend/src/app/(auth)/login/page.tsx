/*
    Renders the login component on the server side.
*/

import { auth } from "@/lib/auth";
import { redirect } from "next/navigation";
import LoginComponent from "../../../components/LoginComponent";

export default async function LoginPage() {
    const session = await auth();

    if (session) {
        redirect("/dashboard");
    }

    return <LoginComponent />;
}
