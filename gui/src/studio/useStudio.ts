/**
 * Studio state — kept separate from the main pipeline store so the
 * image-dataset world (classes, images, annotations, suggestions, trained
 * YOLO26 models) doesn't tangle with Teacher/Student runs.
 *
 * Annotations and suggestions are cached per image; the server is the
 * source of truth and every mutation writes through immediately.
 */

import { create } from "zustand";
import * as api from "./api";
import type {
  Annotation,
  Catalog,
  DatasetStats,
  ModelSpec,
  StudioClass,
  StudioImage,
  SuggestionSet,
  TrainedModel,
} from "./types";

export type StudioTab = "label" | "train" | "test" | "pallet";

interface Toast {
  id: number;
  kind: "info" | "error" | "success";
  text: string;
}

interface StudioStore {
  projectId: string | null;
  tab: StudioTab;
  catalog: Catalog | null;
  classes: StudioClass[];
  images: StudioImage[];
  stats: DatasetStats | null;
  currentImageId: string | null;
  activeClassId: number | null;
  annotations: Record<string, Annotation[]>;
  suggestions: Record<string, SuggestionSet>;
  models: TrainedModel[];
  busy: string | null;
  toasts: Toast[];
  /** Model the Test tab should preselect (set from Train → "Test it"). */
  testModel: ModelSpec | null;

  setTab: (t: StudioTab) => void;
  load: (pid: string) => Promise<void>;
  refresh: () => Promise<void>;
  refreshModels: () => Promise<void>;
  selectImage: (iid: string | null) => Promise<void>;
  loadImageData: (iid: string) => Promise<void>;
  setActiveClass: (id: number | null) => void;
  saveClasses: (classes: Partial<StudioClass>[]) => Promise<void>;
  addClass: (name: string) => Promise<StudioClass | null>;
  setAnnotations: (iid: string, anns: Partial<Annotation>[]) => Promise<void>;
  applyImage: (img: StudioImage) => void;
  applyImages: (imgs: StudioImage[]) => void;
  setSuggestions: (iid: string, s: SuggestionSet) => void;
  setTestModel: (m: ModelSpec | null) => void;
  run: <T>(label: string, fn: () => Promise<T>) => Promise<T | undefined>;
  toast: (text: string, kind?: Toast["kind"]) => void;
  dismissToast: (id: number) => void;
}

let toastSeq = 1;

export const useStudio = create<StudioStore>((set, get) => ({
  projectId: null,
  tab: "label",
  catalog: null,
  classes: [],
  images: [],
  stats: null,
  currentImageId: null,
  activeClassId: null,
  annotations: {},
  suggestions: {},
  models: [],
  busy: null,
  toasts: [],
  testModel: null,

  setTab: (t) => set({ tab: t }),

  load: async (pid) => {
    if (get().projectId !== pid) {
      set({
        projectId: pid,
        classes: [],
        images: [],
        stats: null,
        currentImageId: null,
        annotations: {},
        suggestions: {},
        models: [],
        activeClassId: null,
      });
    }
    if (!get().catalog) {
      api.fetchCatalog().then((catalog) => set({ catalog })).catch((e) => get().toast(String(e), "error"));
    }
    await get().refresh();
    await get().refreshModels();
    const { images, currentImageId } = get();
    if (!currentImageId && images.length) await get().selectImage(images[0].id);
  },

  refresh: async () => {
    const pid = get().projectId;
    if (!pid) return;
    const st = await api.fetchStudio(pid);
    const active = get().activeClassId;
    set({
      classes: st.classes,
      images: st.images,
      stats: st.stats,
      activeClassId:
        active !== null && st.classes.some((c) => c.id === active) ? active : st.classes[0]?.id ?? null,
    });
  },

  refreshModels: async () => {
    const pid = get().projectId;
    if (!pid) return;
    const { models } = await api.fetchModels(pid);
    set({ models });
  },

  selectImage: async (iid) => {
    set({ currentImageId: iid });
    if (iid) await get().loadImageData(iid);
  },

  loadImageData: async (iid) => {
    const pid = get().projectId;
    if (!pid) return;
    const [a, s] = await Promise.all([api.fetchAnnotations(pid, iid), api.fetchSuggestions(pid, iid)]);
    set((st) => ({
      annotations: { ...st.annotations, [iid]: a.annotations },
      suggestions: { ...st.suggestions, [iid]: s },
    }));
  },

  setActiveClass: (id) => set({ activeClassId: id }),

  saveClasses: async (classes) => {
    const pid = get().projectId;
    if (!pid) return;
    const out = await api.putClasses(pid, classes);
    set({ classes: out.classes, images: out.images });
    // Class removal can delete annotations server-side; drop stale caches.
    const cur = get().currentImageId;
    set({ annotations: {} });
    if (cur) await get().loadImageData(cur);
    await get().refresh();
  },

  addClass: async (name) => {
    const trimmed = name.trim();
    if (!trimmed) return null;
    const existing = get().classes.find((c) => c.name.toLowerCase() === trimmed.toLowerCase());
    if (existing) {
      set({ activeClassId: existing.id });
      return existing;
    }
    await get().saveClasses([...get().classes, { name: trimmed }]);
    const created = get().classes.find((c) => c.name.toLowerCase() === trimmed.toLowerCase()) ?? null;
    if (created) set({ activeClassId: created.id });
    return created;
  },

  setAnnotations: async (iid, anns) => {
    const pid = get().projectId;
    if (!pid) return;
    const out = await api.putAnnotations(pid, iid, anns);
    set((st) => ({ annotations: { ...st.annotations, [iid]: out.annotations } }));
    get().applyImage(out.image);
  },

  applyImage: (img) =>
    set((st) => ({ images: st.images.map((i) => (i.id === img.id ? img : i)) })),

  applyImages: (imgs) => set({ images: imgs }),

  setSuggestions: (iid, s) => set((st) => ({ suggestions: { ...st.suggestions, [iid]: s } })),

  setTestModel: (m) => set({ testModel: m }),

  run: async (label, fn) => {
    set({ busy: label });
    try {
      return await fn();
    } catch (e) {
      get().toast(e instanceof Error ? e.message : String(e), "error");
      return undefined;
    } finally {
      set({ busy: null });
    }
  },

  toast: (text, kind = "info") => {
    const id = toastSeq++;
    set((st) => ({ toasts: [...st.toasts, { id, kind, text }] }));
    window.setTimeout(() => get().dismissToast(id), kind === "error" ? 9000 : 4500);
  },

  dismissToast: (id) => set((st) => ({ toasts: st.toasts.filter((t) => t.id !== id) })),
}));

export function classColor(classes: StudioClass[], id: number | null | undefined): string {
  return classes.find((c) => c.id === id)?.color ?? "#9ca3af";
}

export function className(classes: StudioClass[], id: number | null | undefined): string {
  return classes.find((c) => c.id === id)?.name ?? "?";
}
