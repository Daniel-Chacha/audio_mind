import { rankProbabilities } from "../rank";
test("sorts desc and flags a single winner", () => {
  const out = rankProbabilities([
    { genre: "pop", prob: 0.1 }, { genre: "rock", prob: 0.7 }, { genre: "jazz", prob: 0.2 },
  ]);
  expect(out.map(o => o.genre)).toEqual(["rock", "jazz", "pop"]);
  expect(out.filter(o => o.win)).toHaveLength(1);
  expect(out[0].win).toBe(true);
});
