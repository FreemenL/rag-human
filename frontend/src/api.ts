import type {
  AuthSession,
  ChatHistory,
  ChatThread,
  InsuranceCategory,
  PlanInterrupt,
  ProductPage,
} from './types';

const BUSINESS_API = '/business-api/api/v1';
const AGENT_API = '/agent-api/api/v1';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BUSINESS_API}${path}`, {
    ...init,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new ApiError(error?.message ?? '请求失败，请稍后重试', response.status);
  }
  return response.json() as Promise<T>;
}

async function authorizedBusinessRequest<T>(
  path: string,
  token: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${BUSINESS_API}${path}`, {
    ...init,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new ApiError(error?.message ?? '璇锋眰澶辫触锛岃绋嶅悗閲嶈瘯', response.status);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

async function agentRequest<T>(
  path: string,
  token: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${AGENT_API}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new ApiError(error?.detail ?? error?.message ?? '请求失败，请稍后重试', response.status);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export function listProducts(
  category: InsuranceCategory | 'all',
  signal?: AbortSignal,
): Promise<ProductPage> {
  const params = new URLSearchParams({ page: '1', page_size: '20' });
  if (category !== 'all') {
    params.set('category', category);
  }
  return request<ProductPage>(`/products?${params}`, { signal });
}

export function login(username: string, password: string): Promise<AuthSession> {
  return request<AuthSession>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  });
}

export function register(
  username: string,
  email: string,
  password: string,
): Promise<AuthSession> {
  return request<AuthSession>('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ username, email, password }),
  });
}

export function refreshSession(): Promise<AuthSession> {
  return request<AuthSession>('/auth/refresh', {
    method: 'POST',
  });
}

export async function logoutSession(): Promise<void> {
  await request<{ ok: boolean }>('/auth/logout', {
    method: 'POST',
  });
}

export function markPlanItemInsured(
  token: string,
  planId: string,
  itemId: string,
): Promise<unknown> {
  return authorizedBusinessRequest<unknown>(
    `/insurance-plans/${encodeURIComponent(planId)}/items/${encodeURIComponent(itemId)}/insured`,
    token,
    { method: 'PATCH' },
  );
}

export function getInsurancePlan(token: string, planId: string): Promise<unknown> {
  return authorizedBusinessRequest<unknown>(
    `/insurance-plans/${encodeURIComponent(planId)}`,
    token,
  );
}

export function listChatThreads(token: string): Promise<ChatThread[]> {
  return agentRequest<ChatThread[]>('/chat-threads', token);
}

export function createChatThread(token: string, title = '保险咨询'): Promise<ChatThread> {
  return agentRequest<ChatThread>('/chat-threads', token, {
    method: 'POST',
    body: JSON.stringify({ title }),
  });
}

export function getChatHistory(token: string, threadId: string): Promise<ChatHistory> {
  return agentRequest<ChatHistory>(`/chat-threads/${threadId}/messages`, token);
}

export function renameChatThread(
  token: string,
  threadId: string,
  title: string,
): Promise<ChatThread> {
  return agentRequest<ChatThread>(`/chat-threads/${threadId}`, token, {
    method: 'PATCH',
    body: JSON.stringify({ title }),
  });
}

export async function deleteChatThread(token: string, threadId: string): Promise<void> {
  await agentRequest<void>(`/chat-threads/${threadId}`, token, {
    method: 'DELETE',
  });
}

export async function streamChat(
  token: string,
  threadId: string,
  input: { message?: string; decision?: Record<string, unknown> },
  handlers: {
    onMessage: (chunk: string) => void;
    onAdditionalInfo?: (info: unknown) => void;
    onInterrupt?: (interrupt: PlanInterrupt) => void;
  },
): Promise<void> {
  const response = await fetch(`${AGENT_API}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ thread_id: threadId, ...input }),
  });
  if (!response.ok || !response.body) {
    const error = await response.json().catch(() => null);
    throw new ApiError(error?.detail ?? error?.message ?? '消息发送失败，请稍后重试', response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split(/\r?\n\r?\n/);
    buffer = parts.pop() ?? '';
    parts.forEach((eventText) => handleSseEvent(eventText, handlers));
  }
  if (buffer.trim()) {
    handleSseEvent(buffer, handlers);
  }
}

function handleSseEvent(
  eventText: string,
  handlers: {
    onMessage: (chunk: string) => void;
    onAdditionalInfo?: (info: unknown) => void;
    onInterrupt?: (interrupt: PlanInterrupt) => void;
  },
) {
  const lines = eventText.split(/\r?\n/);
  const event = lines.find((line) => line.startsWith('event:'))?.slice(6).trim();
  const data = lines
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())
    .join('\n');
  if (!data) return;
  if (event === 'message') {
    handlers.onMessage(parseSseMessage(data));
    return;
  }
  if (event === 'interrupt') {
    const interrupt = parseInterrupt(data);
    if (interrupt) handlers.onInterrupt?.(interrupt);
    return;
  }
  if (event === 'additional_info') {
    handlers.onAdditionalInfo?.(parseJsonLike(data) ?? data);
  }
}

function parseSseMessage(data: string): string {
  try {
    const value = JSON.parse(data) as unknown;
    return typeof value === 'string' ? value : JSON.stringify(value);
  } catch {
    return data;
  }
}

function parseInterrupt(data: string): PlanInterrupt | null {
  const parsed = parseJsonLike(data);
  const value = Array.isArray(parsed) ? parsed[0] : parsed;
  if (!value || typeof value !== 'object') return null;
  const candidate = value as Partial<PlanInterrupt>;
  if (!('plan' in candidate) || typeof candidate.confirm_info !== 'string') return null;
  return {
    plan: candidate.plan,
    confirm_info: candidate.confirm_info,
    action: Array.isArray(candidate.action)
      ? candidate.action.filter((item): item is 'approve' | 'reject' =>
          item === 'approve' || item === 'reject',
        )
      : ['approve', 'reject'],
  };
}

function parseJsonLike(data: string): unknown {
  try {
    return JSON.parse(data);
  } catch {
    return null;
  }
}
