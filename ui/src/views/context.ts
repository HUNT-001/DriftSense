import type { Api } from '../api/types';
import type { AppConfig } from '../config';
import type { WorkerInfo } from '../types/contracts';

export interface AppContext {
  api: Api;
  config: AppConfig;
  /** Polite screen-reader announcement (ARIA live region). */
  announce(message: string): void;
  navigate(hash: string): void;
  /** Report the worker that ran the current job (footer shows "COOL worker: yes/no"). */
  setWorker(worker: WorkerInfo | null | undefined): void;
}

export interface View {
  el: HTMLElement;
  /** Heading to focus after navigation. */
  heading?: HTMLElement;
  dispose?(): void;
}
