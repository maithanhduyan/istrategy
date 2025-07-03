import express from 'express';
import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import compression from 'compression';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import sqlite3 from 'sqlite3';

// Types
interface User {
  id: number;
  email: string;
  password_hash: string;
  api_key: string;
  plan: string;
  created_at: string;
}

interface AuthenticatedRequest extends express.Request {
  user?: User;
}

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server });

// Database setup
const db = new sqlite3.Database('./monitoring.db');

// Database helper functions
function dbRun(sql: string, params: any[] = []): Promise<any> {
  return new Promise((resolve, reject) => {
    db.run(sql, params, function(err) {
      if (err) reject(err);
      else resolve({ lastID: this.lastID, changes: this.changes });
    });
  });
}

function dbGet(sql: string, params: any[] = []): Promise<any> {
  return new Promise((resolve, reject) => {
    db.get(sql, params, (err, row) => {
      if (err) reject(err);
      else resolve(row);
    });
  });
}

function dbAll(sql: string, params: any[] = []): Promise<any[]> {
  return new Promise((resolve, reject) => {
    db.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows);
    });
  });
}

// Initialize database tables
async function initDatabase() {
  await dbRun(`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      email TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      api_key TEXT UNIQUE NOT NULL,
      plan TEXT DEFAULT 'free',
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  await dbRun(`
    CREATE TABLE IF NOT EXISTS monitoring_sessions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER,
      website_url TEXT NOT NULL,
      session_id TEXT UNIQUE NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users (id)
    )
  `);

  await dbRun(`
    CREATE TABLE IF NOT EXISTS error_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      error_message TEXT NOT NULL,
      stack_trace TEXT,
      url TEXT,
      line_number INTEGER,
      column_number INTEGER,
      user_agent TEXT,
      client_ip TEXT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (session_id) REFERENCES monitoring_sessions (session_id)
    )
  `);

  await dbRun(`
    CREATE TABLE IF NOT EXISTS performance_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      page_load_time REAL,
      dom_content_loaded REAL,
      memory_used INTEGER,
      memory_total INTEGER,
      resource_count INTEGER,
      client_ip TEXT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (session_id) REFERENCES monitoring_sessions (session_id)
    )
  `);

  await dbRun(`
    CREATE TABLE IF NOT EXISTS user_actions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      action_type TEXT NOT NULL,
      element TEXT,
      additional_data TEXT,
      client_ip TEXT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (session_id) REFERENCES monitoring_sessions (session_id)
    )
  `);

  console.log('✅ Database initialized successfully');
}

// In-memory logs storage (for real-time dashboard)
const logs = {
  errors: [] as any[],
  warnings: [] as any[],
  info: [] as any[],
  performance: [] as any[],
  userActions: [] as any[]
};

// JWT Secret
const JWT_SECRET = process.env.JWT_SECRET || 'your-super-secret-jwt-key-change-in-production';

// Middleware
app.use(helmet({
  contentSecurityPolicy: false // Allow for development
}));
app.use(cors());
app.use(compression());
app.use(morgan('combined'));
app.use(express.json({ limit: '10mb' }));
app.use(express.static('public'));

// Authentication middleware
async function authenticateAPI(req: AuthenticatedRequest, res: express.Response, next: express.NextFunction) {
  const apiKey = req.headers['x-api-key'] || req.query.apiKey;
  
  if (!apiKey) {
    return res.status(401).json({ error: 'API key required' });
  }

  try {
    const user = await dbGet('SELECT * FROM users WHERE api_key = ?', [apiKey]) as User;
    if (!user) {
      return res.status(401).json({ error: 'Invalid API key' });
    }
    
    req.user = user;
    next();
  } catch (error) {
    res.status(500).json({ error: 'Authentication failed' });
  }
}

// Generate API key
function generateApiKey(): string {
  return 'mk_' + Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
}

// Routes
app.get('/', (req, res) => {
  res.json({
    message: 'NEAT Monitoring Server',
    version: '2.0.0',
    features: [
      'Real-time error monitoring',
      'Performance tracking',
      'User behavior analytics',
      'WebSocket communications',
      'REST API access',
      'Authentication & API keys'
    ],
    endpoints: {
      'GET /': 'Server information',
      'POST /auth/register': 'Register new account',
      'POST /auth/login': 'Login to account',
      'GET /api/dashboard': 'Get monitoring dashboard data',
      'GET /api/errors': 'Get error logs',
      'GET /api/performance': 'Get performance data',
      'GET /api/actions': 'Get user actions',
      'WebSocket': 'ws://localhost:3000 for real-time monitoring'
    },
    timestamp: new Date().toISOString()
  });
});

// Auth endpoints
app.post('/auth/register', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password required' });
    }

    // Check if user exists
    const existingUser = await dbGet('SELECT id FROM users WHERE email = ?', [email]);
    if (existingUser) {
      return res.status(409).json({ error: 'Email already registered' });
    }

    // Hash password and create user
    const passwordHash = await bcrypt.hash(password, 10);
    const apiKey = generateApiKey();
    
    const result = await dbRun(
      'INSERT INTO users (email, password_hash, api_key) VALUES (?, ?, ?)',
      [email, passwordHash, apiKey]
    );

    res.status(201).json({
      success: true,
      message: 'Account created successfully',
      apiKey: apiKey,
      userId: (result as any).lastID
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ error: 'Failed to create account' });
  }
});

app.post('/auth/login', async (req: express.Request, res: express.Response) => {
  try {
    const { email, password } = req.body;
    
    const user = await dbGet('SELECT * FROM users WHERE email = ?', [email]) as User;
    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const isValidPassword = await bcrypt.compare(password, user.password_hash);
    if (!isValidPassword) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const token = jwt.sign({ userId: user.id, email: user.email }, JWT_SECRET, { expiresIn: '24h' });

    res.json({
      success: true,
      token: token,
      apiKey: user.api_key,
      user: {
        id: user.id,
        email: user.email,
        plan: user.plan,
        createdAt: user.created_at
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Login failed' });
  }
});

// API endpoints (protected)
app.get('/api/dashboard', authenticateAPI, async (req: AuthenticatedRequest, res: express.Response) => {
  try {
    const userId = req.user!.id;
    
    // Get stats for user's sessions
    const errorCount = await dbGet(
      `SELECT COUNT(*) as count FROM error_logs el 
       JOIN monitoring_sessions ms ON el.session_id = ms.session_id 
       WHERE ms.user_id = ?`,
      [userId]
    ) as { count: number };

    const sessionCount = await dbGet(
      'SELECT COUNT(*) as count FROM monitoring_sessions WHERE user_id = ?',
      [userId]
    ) as { count: number };

    const recentErrors = await dbAll(
      `SELECT el.*, ms.website_url FROM error_logs el 
       JOIN monitoring_sessions ms ON el.session_id = ms.session_id 
       WHERE ms.user_id = ? 
       ORDER BY el.timestamp DESC LIMIT 10`,
      [userId]
    );

    const performanceData = await dbAll(
      `SELECT pl.*, ms.website_url FROM performance_logs pl 
       JOIN monitoring_sessions ms ON pl.session_id = ms.session_id 
       WHERE ms.user_id = ? 
       ORDER BY pl.timestamp DESC LIMIT 10`,
      [userId]
    );

    res.json({
      user: req.user,
      stats: {
        totalErrors: errorCount.count,
        totalSessions: sessionCount.count,
        planLimits: {
          free: { errors: 1000, sessions: 10 },
          pro: { errors: 50000, sessions: 100 },
          enterprise: { errors: -1, sessions: -1 }
        }
      },
      recentErrors,
      performanceData,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Dashboard error:', error);
    res.status(500).json({ error: 'Failed to load dashboard' });
  }
});

app.get('/api/errors', authenticateAPI, async (req: AuthenticatedRequest, res: express.Response) => {
  try {
    const userId = req.user!.id;
    const limit = parseInt(req.query.limit as string) || 50;
    const offset = parseInt(req.query.offset as string) || 0;

    const errors = await dbAll(
      `SELECT el.*, ms.website_url FROM error_logs el 
       JOIN monitoring_sessions ms ON el.session_id = ms.session_id 
       WHERE ms.user_id = ? 
       ORDER BY el.timestamp DESC LIMIT ? OFFSET ?`,
      [userId, limit, offset]
    );

    res.json({
      errors,
      pagination: { limit, offset, hasMore: errors.length === limit }
    });
  } catch (error) {
    console.error('Errors API error:', error);
    res.status(500).json({ error: 'Failed to fetch errors' });
  }
});

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    database: 'connected',
    timestamp: new Date().toISOString()
  });
});

// WebSocket connection handling with authentication
wss.on('connection', async (ws: WebSocket, request) => {
  const clientIP = request.socket.remoteAddress || 'unknown';
  const url = new URL(request.url!, `http://${request.headers.host}`);
  const apiKey = url.searchParams.get('apiKey');
  const websiteUrl = url.searchParams.get('website');

  let authenticatedUser = null;
  let sessionId = null;

  // Authenticate WebSocket connection
  if (apiKey) {
    try {
      authenticatedUser = await dbGet('SELECT * FROM users WHERE api_key = ?', [apiKey]);
      if (authenticatedUser && websiteUrl) {
        // Create monitoring session
        sessionId = 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        await dbRun(
          'INSERT INTO monitoring_sessions (user_id, website_url, session_id) VALUES (?, ?, ?)',
          [authenticatedUser.id, websiteUrl, sessionId]
        );
        console.log(`🔗 Authenticated WebSocket connection from ${clientIP} for ${websiteUrl}`);
      }
    } catch (error) {
      console.error('WebSocket auth error:', error);
    }
  }

  if (!authenticatedUser) {
    console.log(`🔗 Unauthenticated WebSocket connection from ${clientIP}`);
  }

  // Send welcome message
  ws.send(JSON.stringify({
    type: 'welcome',
    message: `Connected to NEAT Monitoring Server`,
    authenticated: !!authenticatedUser,
    sessionId: sessionId,
    timestamp: new Date().toISOString()
  }));

  // Handle incoming messages
  ws.on('message', async (data: Buffer) => {
    try {
      const message = JSON.parse(data.toString());
      console.log('📨 Received:', message.type, 'from', clientIP);

      // Store in database if authenticated
      if (authenticatedUser && sessionId) {
        await storeMessageInDB(message, sessionId, clientIP);
      }

      // Store in memory for real-time dashboard
      storeMessageInMemory(message, clientIP);

      // Handle different message types
      switch (message.type) {
        case 'error':
          handleClientError(message, clientIP);
          break;
        case 'warning':
          handleClientWarning(message, clientIP);
          break;
        case 'performance':
          handlePerformanceData(message, clientIP);
          break;
        case 'user_action':
          handleUserAction(message, clientIP);
          break;
        default:
          // Echo back for regular messages
          ws.send(JSON.stringify({
            type: 'echo',
            original: message,
            server: 'NEAT Monitoring Server v2.0',
            timestamp: new Date().toISOString()
          }));
          break;
      }

    } catch (error) {
      console.error('❌ Error processing message:', error);
      ws.send(JSON.stringify({
        type: 'error',
        message: 'Invalid message format',
        timestamp: new Date().toISOString()
      }));
    }
  });

  ws.on('close', (code, reason) => {
    console.log(`🔌 WebSocket disconnected: ${code} - ${reason}`);
  });

  // Send periodic ping
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'ping',
        timestamp: new Date().toISOString()
      }));
    } else {
      clearInterval(pingInterval);
    }
  }, 30000);
});

// Database storage functions
async function storeMessageInDB(message: any, sessionId: string, clientIP: string) {
  try {
    switch (message.type) {
      case 'error':
        await dbRun(
          `INSERT INTO error_logs (session_id, error_message, stack_trace, url, line_number, column_number, user_agent, client_ip) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
          [
            sessionId,
            message.error || message.message,
            message.stack,
            message.url,
            message.line,
            message.column,
            message.userAgent,
            clientIP
          ]
        );
        break;

      case 'performance':
        await dbRun(
          `INSERT INTO performance_logs (session_id, page_load_time, dom_content_loaded, memory_used, memory_total, resource_count, client_ip) 
           VALUES (?, ?, ?, ?, ?, ?, ?)`,
          [
            sessionId,
            message.metrics?.pageLoadTime,
            message.metrics?.domContentLoaded,
            message.metrics?.memoryUsage?.used,
            message.metrics?.memoryUsage?.total,
            message.metrics?.resourceCount,
            clientIP
          ]
        );
        break;

      case 'user_action':
        await dbRun(
          `INSERT INTO user_actions (session_id, action_type, element, additional_data, client_ip) 
           VALUES (?, ?, ?, ?, ?)`,
          [
            sessionId,
            message.action,
            message.element,
            JSON.stringify(message.data || {}),
            clientIP
          ]
        );
        break;
    }
  } catch (error) {
    console.error('Database storage error:', error);
  }
}

// Memory storage functions (existing functions with minor updates)
function storeMessageInMemory(message: any, clientIP: string) {
  const timestamp = new Date().toISOString();
  const entry = { ...message, clientIP, timestamp };

  switch (message.type) {
    case 'error':
      logs.errors.push(entry);
      if (logs.errors.length > 100) logs.errors = logs.errors.slice(-100);
      break;
    case 'warning':
      logs.warnings.push(entry);
      if (logs.warnings.length > 100) logs.warnings = logs.warnings.slice(-100);
      break;
    case 'performance':
      logs.performance.push(entry);
      if (logs.performance.length > 50) logs.performance = logs.performance.slice(-50);
      break;
    case 'user_action':
      logs.userActions.push(entry);
      if (logs.userActions.length > 200) logs.userActions = logs.userActions.slice(-200);
      break;
  }
}

// Handler functions (simplified)
function handleClientError(message: any, clientIP: string) {
  console.error('🚨 CLIENT ERROR:', message.error || message.message, 'from', clientIP);
}

function handleClientWarning(message: any, clientIP: string) {
  console.warn('⚠️ CLIENT WARNING:', message.warning || message.message, 'from', clientIP);
}

function handlePerformanceData(message: any, clientIP: string) {
  console.log('📊 PERFORMANCE:', `Load: ${message.metrics?.pageLoadTime?.toFixed(2)}ms`, 'from', clientIP);
}

function handleUserAction(message: any, clientIP: string) {
  console.log('👤 USER ACTION:', message.action, 'on', message.element, 'from', clientIP);
}

// Legacy logs endpoint for backward compatibility
app.get('/logs', (req, res) => {
  res.json({
    server: 'NEAT Monitoring Server v2.0',
    logs: {
      summary: {
        totalErrors: logs.errors.length,
        totalWarnings: logs.warnings.length,
        totalPerformance: logs.performance.length,
        totalUserActions: logs.userActions.length
      },
      recent: {
        errors: logs.errors.slice(-10),
        warnings: logs.warnings.slice(-10),
        performance: logs.performance.slice(-5),
        userActions: logs.userActions.slice(-20)
      }
    },
    clientCount: wss.clients.size,
    timestamp: new Date().toISOString()
  });
});

// Server startup
const PORT = process.env.PORT || 3000;

async function startServer() {
  try {
    await initDatabase();
    
    server.listen(PORT, () => {
      console.log('🚀 NEAT Monitoring Server v2.0 starting...');
      console.log(`📡 HTTP Server: http://localhost:${PORT}`);
      console.log(`🔌 WebSocket: ws://localhost:${PORT}?apiKey=YOUR_API_KEY&website=https://yoursite.com`);
      console.log(`🔐 Authentication: Enabled`);
      console.log(`💾 Database: SQLite (./monitoring.db)`);
      console.log(`👥 Connected clients: ${wss.clients.size}`);
      console.log('✅ Server is ready for production!');
    });
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('🛑 Received SIGTERM, shutting down gracefully...');
  server.close(() => {
    db.close();
    console.log('✅ Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  console.log('🛑 Received SIGINT, shutting down gracefully...');
  server.close(() => {
    db.close();
    console.log('✅ Server closed');
    process.exit(0);
  });
});

startServer();

export default app;
