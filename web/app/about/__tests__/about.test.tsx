import { render, screen } from "@testing-library/react";
import About from "../page";

test("renders the about page with grounded product facts", () => {
  render(<About />);
  expect(
    screen.getByRole("heading", { name: /runs entirely in your browser/i }),
  ).toBeInTheDocument();
  // Real, verifiable figures — guards against them silently being edited out.
  expect(screen.getByText("85.3%")).toBeInTheDocument();
  expect(screen.getByText("78.3%")).toBeInTheDocument();
  expect(screen.getByText(/Chance is 10%/)).toBeInTheDocument();
  // Navigation back to the analyzer exists.
  expect(screen.getByRole("link", { name: /analyzer/i })).toHaveAttribute("href", "/");
});
