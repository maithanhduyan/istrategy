export interface User {
  id: number;
  username: string;
  created_at: string;
}

export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

export interface ApiKeyResponse {
  api_key: string;
}

export interface ServerStatus {
  status: string;
  message: string;
}

export interface DatabaseInfo {
  database: string;
  tables: string[];
  users_count: number;
}
