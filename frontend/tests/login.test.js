import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import LoginComponent from "@/components/LoginComponent";
import { signIn } from "next-auth/react";

// Reusable push mock
const push = jest.fn();

// Mock useRouter
jest.mock("next/navigation", () => ({
    useRouter: () => ({
      push,
      replace: jest.fn(),
      refresh: jest.fn(),
      prefetch: jest.fn(),
    }),
}));

// Mock signIn from next-auth
jest.mock("next-auth/react", () => ({
  signIn: jest.fn(),
}));

// Global fetch mock
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ user: { username: "testuser" } }),
  })
);

describe("LoginComponent", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

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

    expect(usernameInput).toHaveValue("testuser");
    expect(passwordInput).toHaveValue("password123");
  });

  test("redirects to dashboard on successful login", async () => {
    render(<LoginComponent />);
  
    fireEvent.change(screen.getByLabelText("Username"), {
      target: { value: "testuser" },
    });
    fireEvent.change(screen.getByLabelText("Password"), {
      target: { value: "password123" },
    });
  
    fireEvent.click(screen.getByRole("button", { name: /login/i }));
  
    await waitFor(() => {
      expect(push).toHaveBeenCalledWith("/dashboard");
    });
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

    await waitFor(() =>
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument()
    );
  });

  test("does not submit if fields are empty", async () => {
    render(<LoginComponent />);

    // wait for hydration to finish
    await waitFor(() => {
        expect(screen.getByRole("button", { name: /login/i })).toBeInTheDocument();
    });

    // trigger submit
    fireEvent.click(screen.getByRole("button", { name: /login/i }));

    // confirm fetch not called
    expect(global.fetch).not.toHaveBeenCalled();

    //wait for alert and assert its contents
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent(/username and password are required/i);
  });

  test("disables login button during submission", async () => {
    render(<LoginComponent />);

    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "testuser" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "password123" } });

    const loginButton = screen.getByRole("button", { name: /login/i });
    fireEvent.click(loginButton);

    expect(loginButton).toBeDisabled();

    await waitFor(() => expect(loginButton).not.toBeDisabled());
  });

  test("clicking Google Sign In calls signIn", () => {
    render(<LoginComponent />);
    const googleButton = screen.getByTestId("google-sign-in");

    fireEvent.click(googleButton);

    expect(signIn).toHaveBeenCalledWith("google", { callbackUrl: "http://localhost:3000/callback" });
  });
});
