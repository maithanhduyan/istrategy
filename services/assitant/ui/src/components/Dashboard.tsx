import React, { useState, useEffect } from 'react';
import { Key, Users, Server, Database, Copy, Check } from 'lucide-react';
import { authService, userService, serverService } from '../services/api';
import type { User, ApiKeyResponse, ServerStatus, DatabaseInfo } from '../types';

interface DashboardProps {
  onLogout: () => void;
}

const Dashboard: React.FC<DashboardProps> = ({ onLogout }) => {
  const [users, setUsers] = useState<User[]>([]);
  const [apiKey, setApiKey] = useState<string>('');
  const [serverStatus, setServerStatus] = useState<ServerStatus>({ status: 'unknown', message: '' });
  const [databaseInfo, setDatabaseInfo] = useState<DatabaseInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      
      // Load all data in parallel
      const [usersData, apiKeyData, statusData, dbData] = await Promise.all([
        userService.getUsers(),
        authService.getApiKey(),
        serverService.getStatus(),
        serverService.getDatabaseInfo()
      ]);

      setUsers(usersData);
      setApiKey(apiKeyData.api_key);
      setServerStatus(statusData);
      setDatabaseInfo(dbData);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const copyApiKey = async () => {
    try {
      await navigator.clipboard.writeText(apiKey);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  const handleLogout = async () => {
    try {
      await authService.logout();
      onLogout();
    } catch (error) {
      console.error('Logout error:', error);
      onLogout(); // Force logout even if API call fails
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Server className="h-8 w-8 text-primary-600 mr-3" />
              <h1 className="text-xl font-semibold text-gray-900">Assistant Dashboard</h1>
            </div>
            <button
              onClick={handleLogout}
              className="btn-secondary"
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* API Key Card */}
          <div className="card">
            <div className="flex items-center mb-4">
              <Key className="h-5 w-5 text-primary-600 mr-2" />
              <h2 className="text-lg font-semibold text-gray-900">X-API-Key</h2>
            </div>
            <div className="space-y-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="flex items-center justify-between">
                  <code className="text-sm font-mono text-gray-800 break-all pr-4">
                    {apiKey || 'Loading...'}
                  </code>
                  <button
                    onClick={copyApiKey}
                    className="flex-shrink-0 p-2 text-gray-500 hover:text-gray-700 transition-colors"
                    disabled={!apiKey}
                  >
                    {copied ? (
                      <Check className="h-4 w-4 text-green-600" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>
              <p className="text-sm text-gray-600">
                Use this API key to authenticate MCP requests to the assistant service.
                Add it as <code className="bg-gray-100 px-1 rounded">X-API-Key</code> header.
              </p>
            </div>
          </div>

          {/* Server Status Card */}
          <div className="card">
            <div className="flex items-center mb-4">
              <Server className="h-5 w-5 text-primary-600 mr-2" />
              <h2 className="text-lg font-semibold text-gray-900">Server Status</h2>
            </div>
            <div className="space-y-3">
              <div className="flex items-center">
                <span className="text-sm text-gray-600 w-20">Status:</span>
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  serverStatus.status === 'running' 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-yellow-100 text-yellow-800'
                }`}>
                  <span className={`w-1.5 h-1.5 rounded-full mr-1.5 ${
                    serverStatus.status === 'running' ? 'bg-green-400' : 'bg-yellow-400'
                  }`}></span>
                  {serverStatus.status}
                </span>
              </div>
              <div className="flex items-center">
                <span className="text-sm text-gray-600 w-20">Message:</span>
                <span className="text-sm text-gray-800">{serverStatus.message}</span>
              </div>
            </div>
          </div>

          {/* Database Info Card */}
          {databaseInfo && (
            <div className="card">
              <div className="flex items-center mb-4">
                <Database className="h-5 w-5 text-primary-600 mr-2" />
                <h2 className="text-lg font-semibold text-gray-900">Database Info</h2>
              </div>
              <div className="space-y-3">
                <div className="flex items-center">
                  <span className="text-sm text-gray-600 w-24">Type:</span>
                  <span className="text-sm text-gray-800">{databaseInfo.database}</span>
                </div>
                <div className="flex items-center">
                  <span className="text-sm text-gray-600 w-24">Tables:</span>
                  <span className="text-sm text-gray-800">{databaseInfo.tables.join(', ')}</span>
                </div>
                <div className="flex items-center">
                  <span className="text-sm text-gray-600 w-24">Users:</span>
                  <span className="text-sm text-gray-800">{databaseInfo.users_count}</span>
                </div>
              </div>
            </div>
          )}

          {/* Users List Card */}
          <div className="card">
            <div className="flex items-center mb-4">
              <Users className="h-5 w-5 text-primary-600 mr-2" />
              <h2 className="text-lg font-semibold text-gray-900">Users ({users.length})</h2>
            </div>
            <div className="space-y-3">
              {users.length > 0 ? (
                users.map((user) => (
                  <div key={user.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div>
                      <div className="font-medium text-gray-900">{user.username}</div>
                      <div className="text-sm text-gray-600">ID: {user.id}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-600">Created</div>
                      <div className="text-xs text-gray-500">
                        {new Date(user.created_at).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-gray-500 text-center py-4">No users found</p>
              )}
            </div>
          </div>
        </div>

        {/* MCP Configuration Example */}
        <div className="mt-8 card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">VS Code MCP Configuration</h2>
          <p className="text-sm text-gray-600 mb-4">
            Add this configuration to your <code className="bg-gray-100 px-1 rounded">.vscode/mcp.json</code> file:
          </p>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`{
  "servers": {
    "assistant": {
      "url": "http://localhost:8001/mcp",
      "headers": {
        "X-API-Key": "${apiKey}"
      }
    }
  }
}`}
            </pre>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
