import { useStore } from "../store";
import { BLOCK_KINDS } from "../types";

export function Palette() {
  const blocks = useStore((s) => s.blocks);
  return (
    <aside className="palette">
      <h2>Blocks</h2>
      <ul>
        {BLOCK_KINDS.map((kind) => {
          const impls = blocks[kind];
          return (
            <li key={kind}>
              <div className="palette-kind">{kind}</div>
              <div className="palette-impls">
                {impls ? impls.join(", ") : "…"}
              </div>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
