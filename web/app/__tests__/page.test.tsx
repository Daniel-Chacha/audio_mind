import { render, screen } from "@testing-library/react";
import Home from "../page";

test("renders the wordmark", () => {
  render(<Home />);
  expect(screen.getByText("AUDIOMIND")).toBeInTheDocument();
});
