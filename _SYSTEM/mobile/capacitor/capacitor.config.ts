import type { CapacitorConfig } from '@capacitor/cli';

const processLike = (globalThis as typeof globalThis & {
  process?: { env?: Record<string, string | undefined> };
}).process;
const serverURL = processLike?.env?.IG_AUTOMATIK_SERVER_URL;

const config: CapacitorConfig = {
  appId: 'com.igautomatik.mobile',
  appName: 'IG-AUTOMATIK',
  webDir: '../web',
  // In the native app, set this to the reachable IG-AUTOMATIK server URL,
  // for example http://192.168.178.50:8787 or a trusted HTTPS URL.
  // Keep it empty during local Capacitor setup if the URL is not known yet.
  server: {
    cleartext: true,
    ...(serverURL ? { url: serverURL } : {}),
  },
};

export default config;
