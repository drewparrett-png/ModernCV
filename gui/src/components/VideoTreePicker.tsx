/**
 * Folder-tree video picker.
 *
 * Replaces the flat `<select>` of every video under data/ with a navigable
 * tree. The flat select breaks down once data/ has hundreds-to-thousands of
 * clips (e.g. the DFL Bundesliga split has ~2k+ shards) — you can't see the
 * folder structure, can't scope by subset, and the dropdown becomes a wall
 * of identical filenames.
 *
 * The server already returns paths relative to the project root (e.g.
 *   "data/DFL Bundesliga Data Shootout/train/A1606b0e6_0/A1606b0e6_0 (12).mp4"
 * ). We build a tree on the client by splitting on "/", so no new API is
 * needed and the picker degrades to "no videos found" when the list is
 * empty.
 *
 * UX choices:
 *   - Folders are collapsed by default. Only the immediate children of the
 *     project root render until the user opens a folder. With ~2k clips
 *     spread across nested folders this keeps initial render cheap.
 *   - Filter input narrows the tree to paths matching the substring; while
 *     filtering, every folder on the matching paths auto-expands so matches
 *     are visible without manual click-through.
 *   - Each folder shows a leaf-count badge so the user knows roughly what
 *     they're about to open.
 *   - Click-outside closes the panel; selecting a file also closes it.
 */

import { useEffect, useMemo, useRef, useState } from "react";

interface TreeNode {
  name: string;
  fullPath: string;
  isFile: boolean;
  /** Cached descendant file count, computed during build. Avoids recomputing
   *  on every row render; for ~2k-leaf trees the recursive walk is cheap
   *  but only when done once. */
  fileCount: number;
  children: TreeNode[];
}

function buildTree(paths: string[]): TreeNode {
  const root: TreeNode = {
    name: "",
    fullPath: "",
    isFile: false,
    fileCount: 0,
    children: [],
  };
  for (const p of paths) {
    const parts = p.split("/").filter((s) => s.length > 0);
    if (parts.length === 0) continue;
    let cursor = root;
    for (let i = 0; i < parts.length; i++) {
      const name = parts[i];
      const isLast = i === parts.length - 1;
      let child = cursor.children.find((c) => c.name === name);
      if (!child) {
        child = {
          name,
          fullPath: parts.slice(0, i + 1).join("/"),
          isFile: isLast,
          fileCount: 0,
          children: [],
        };
        cursor.children.push(child);
      }
      cursor = child;
    }
  }
  // Compute fileCount + sort: dirs first, then alphabetical w/ natural
  // ordering so "test (2).mp4" comes before "test (10).mp4".
  const finalize = (n: TreeNode): number => {
    if (n.isFile) {
      n.fileCount = 1;
      return 1;
    }
    let total = 0;
    for (const c of n.children) total += finalize(c);
    n.fileCount = total;
    n.children.sort((a, b) => {
      if (a.isFile !== b.isFile) return a.isFile ? 1 : -1;
      return a.name.localeCompare(b.name, undefined, { numeric: true });
    });
    return total;
  };
  finalize(root);
  return root;
}

/** Walk all ancestors of `path` and add their fullPaths to `set`, so the
 *  tree auto-expands to reveal a target node. */
function expandAncestors(path: string, set: Set<string>): void {
  if (!path) return;
  const parts = path.split("/").filter((s) => s.length > 0);
  for (let i = 1; i < parts.length; i++) {
    set.add(parts.slice(0, i).join("/"));
  }
}

export function VideoTreePicker({
  videos,
  selected,
  onSelect,
  emptyHint = "(no videos in data/ — add some clips and refresh)",
}: {
  videos: string[];
  selected: string;
  onSelect: (path: string) => void;
  emptyHint?: string;
}) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  // Persist expansion across opens so re-opening lands you where you were.
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const containerRef = useRef<HTMLDivElement>(null);

  // Click-outside to dismiss.
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  // When the picker opens, auto-expand the path of the current selection so
  // the user can see where they are. Don't collapse what they had open.
  useEffect(() => {
    if (open && selected) {
      setExpanded((prev) => {
        const next = new Set(prev);
        expandAncestors(selected, next);
        return next;
      });
    }
  }, [open, selected]);

  const tree = useMemo(() => buildTree(videos), [videos]);

  const filterTrim = filter.trim().toLowerCase();
  const filteredTree = useMemo(() => {
    if (filterTrim === "") return tree;
    return buildTree(
      videos.filter((p) => p.toLowerCase().includes(filterTrim)),
    );
  }, [tree, videos, filterTrim]);

  // While filtering, every folder on the surviving paths auto-expands.
  const expandedForRender = useMemo(() => {
    if (filterTrim === "") return expanded;
    const all = new Set<string>(expanded);
    const walk = (n: TreeNode) => {
      if (!n.isFile) {
        all.add(n.fullPath);
        n.children.forEach(walk);
      }
    };
    walk(filteredTree);
    return all;
  }, [filteredTree, filterTrim, expanded]);

  const toggle = (path: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const select = (path: string) => {
    onSelect(path);
    setOpen(false);
  };

  const triggerLabel = selected
    ? selected
    : videos.length === 0
      ? emptyHint
      : `Pick a video — ${videos.length} available`;

  return (
    <div className="video-picker" ref={containerRef}>
      <button
        type="button"
        className={`video-picker-trigger ${selected ? "has-value" : ""}`}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="video-picker-trigger-label" title={triggerLabel}>
          {triggerLabel}
        </span>
        <span className="video-picker-trigger-chevron">
          {open ? "▴" : "▾"}
        </span>
      </button>
      {open && (
        <div className="video-picker-panel" role="dialog">
          <div className="video-picker-toolbar">
            <input
              type="text"
              className="video-picker-filter"
              placeholder="Filter — substring match anywhere in path"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              autoFocus
            />
            {selected && (
              <button
                type="button"
                className="video-picker-clear"
                onClick={() => select("")}
                title="Clear selection"
              >
                Clear
              </button>
            )}
          </div>
          <div className="video-picker-tree">
            {filteredTree.children.length === 0 ? (
              <div className="video-picker-empty">
                {videos.length === 0
                  ? "No videos in data/. Drop some clips there and refresh."
                  : "No matches for that filter."}
              </div>
            ) : (
              <TreeChildren
                nodes={filteredTree.children}
                depth={0}
                expanded={expandedForRender}
                toggle={toggle}
                onSelect={select}
                selected={selected}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function TreeChildren({
  nodes,
  depth,
  expanded,
  toggle,
  onSelect,
  selected,
}: {
  nodes: TreeNode[];
  depth: number;
  expanded: Set<string>;
  toggle: (path: string) => void;
  onSelect: (path: string) => void;
  selected: string;
}) {
  return (
    <ul className="video-picker-list" role={depth === 0 ? "tree" : "group"}>
      {nodes.map((n) => (
        <TreeRow
          key={n.fullPath}
          node={n}
          depth={depth}
          expanded={expanded}
          toggle={toggle}
          onSelect={onSelect}
          selected={selected}
        />
      ))}
    </ul>
  );
}

function TreeRow({
  node,
  depth,
  expanded,
  toggle,
  onSelect,
  selected,
}: {
  node: TreeNode;
  depth: number;
  expanded: Set<string>;
  toggle: (path: string) => void;
  onSelect: (path: string) => void;
  selected: string;
}) {
  const isOpen = expanded.has(node.fullPath);
  const isSelected = node.isFile && node.fullPath === selected;
  // Indent each level by a fixed amount; chevron is rendered inline so the
  // file/dir name lines up regardless of whether the row has children.
  const indent: React.CSSProperties = { paddingLeft: 8 + depth * 14 };

  if (node.isFile) {
    return (
      <li role="treeitem" aria-selected={isSelected}>
        <button
          type="button"
          className={`video-picker-row file ${isSelected ? "selected" : ""}`}
          style={indent}
          onClick={() => onSelect(node.fullPath)}
          title={node.fullPath}
        >
          <span className="video-picker-bullet">·</span>
          <span className="video-picker-name">{node.name}</span>
        </button>
      </li>
    );
  }
  return (
    <li role="treeitem" aria-expanded={isOpen}>
      <button
        type="button"
        className="video-picker-row dir"
        style={indent}
        onClick={() => toggle(node.fullPath)}
      >
        <span className="video-picker-chevron">{isOpen ? "▾" : "▸"}</span>
        <span className="video-picker-name">{node.name}</span>
        <span className="video-picker-count">{node.fileCount}</span>
      </button>
      {isOpen && node.children.length > 0 && (
        <TreeChildren
          nodes={node.children}
          depth={depth + 1}
          expanded={expanded}
          toggle={toggle}
          onSelect={onSelect}
          selected={selected}
        />
      )}
    </li>
  );
}
