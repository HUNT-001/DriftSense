import './styles.css';
import { mountApp } from './app';
import { createApi, loadConfig } from './config';

async function boot() {
  const root = document.getElementById('app');
  if (!root) return;
  const config = await loadConfig();
  mountApp(root, createApi(config), config);
}

void boot();
