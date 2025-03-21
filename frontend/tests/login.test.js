import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import LoginComponent from "@/components/LoginComponent";

// Mock the fetch API response
global.fetch = jest.fn(() =>
    Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ user: { username: "testuser" } }),
    })
);

describe("LoginComponent", () => {
    test("renders login form with username and password fields", () => {
        render(<LoginComponent />);

        expect(screen.getByLabelText("Username")).toBeInTheDocument();
        expect(screen.getByLabelText("Password")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: /login/i })).toBeInTheDocument();
        expect(screen.getByTestId("google-sign-in")).toBeInTheDocument();
    });

    test("allows the user to type in the username and password fields", () => {
        render(<LoginComponent />);

        const usernameInput = screen.getByLabelText("Username");
        const passwordInput = screen.getByLabelText("Password");

        fireEvent.change(usernameInput, { target: { value: "testuser" } });
        fireEvent.change(passwordInput, { target: { value: "password123" } });

        expect(usernameInput.value).toBe("testuser");
        expect(passwordInput.value).toBe("password123");
    });

    test("displays success message on successful login", async () => {
        render(<LoginComponent />);

        fireEvent.change(screen.getByLabelText("Username"), { target: { value: "testuser" } });
        fireEvent.change(screen.getByLabelText("Password"), { target: { value: "password123" } });

        fireEvent.click(screen.getByRole("button", { name: /login/i }));

        await waitFor(() => expect(screen.getByText(/welcome, testuser!/i)).toBeInTheDocument());
    });

    test("displays error message on failed login", async () => {
        global.fetch.mockImplementationOnce(() =>
            Promise.resolve({
                ok: false,
                json: () => Promise.resolve({ error: "Invalid credentials" }),
            })
        );

        render(<LoginComponent />);

        fireEvent.change(screen.getByLabelText("Username"), { target: { value: "wronguser" } });
        fireEvent.change(screen.getByLabelText("Password"), { target: { value: "wrongpassword" } });

        fireEvent.click(screen.getByRole("button", { name: /login/i }));

        await waitFor(() => expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument());
    });
});
