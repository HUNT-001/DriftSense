import './styles.css';
import { mountApp } from './app';
import { createApi, loadConfig, resolveConfig } from './config';

async function boot() {
  const root = document.getElementById('app');
  if (!root) return;
  const wait = document.createElement('p');
  wait.className = 'boot-wait';
  wait.setAttribute('role', 'status');
  wait.textContent = 'Connecting to the DriftSense API…';
  root.append(wait);
  const config = await resolveConfig(await loadConfig());
  mountApp(root, createApi(config), config);
}

void boot();
