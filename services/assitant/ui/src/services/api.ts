import axios from 'axios';
import type { 
  LoginCredentials, 
  LoginResponse, 
  ApiKeyResponse, 
  User,
  ServerStatus,
  DatabaseInfo 
} from '../types';

// Configure axios base URL
const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json'
  }
});

// Add token to requests if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const authService = {
  async login(credentials: LoginCredentials): Promise<LoginResponse> {
    const formData = new FormData();
    formData.append('username', credentials.username);
    formData.append('password', credentials.password);
    
    const response = await api.post('/auth/login', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  },

  async logout(): Promise<void> {
    await api.post('/auth/logout');
    localStorage.removeItem('access_token');
  },

  async getApiKey(): Promise<ApiKeyResponse> {
    const response = await api.get('/auth/api-key');
    return response.data;
  }
};

export const userService = {
  async getUsers(): Promise<User[]> {
    const response = await api.get('/users');
    return response.data.users;
  }
};

export const serverService = {
  async getStatus(): Promise<ServerStatus> {
    const response = await api.get('/health');
    return {
      status: 'running',
      message: response.data.message || 'Server is healthy'
    };
  },

  async getDatabaseInfo(): Promise<DatabaseInfo> {
    // Mock database info since we don't have a direct endpoint
    return {
      database: 'SQLite',
      tables: ['users'],
      users_count: 1
    };
  }
};

export default api;
