import express from 'express';
import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server });

// In-memory logs storage (in production, use proper logging service)
const logs = {
  errors: [] as any[],
  warnings: [] as any[],
  info: [] as any[],
  performance: [] as any[],
  userActions: [] as any[]
};

// Middleware
app.use(express.json());
app.use(express.static('public'));

// Main endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'Welcome to NEAT Server',
    timestamp: new Date().toISOString(),
    endpoints: {
      http: 'GET /',
      websocket: 'ws://localhost:3000'
    },
    status: 'running'
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    timestamp: new Date().toISOString()
  });
});

// Logs endpoint for monitoring
app.get('/logs', (req, res) => {
  res.json({
    server: 'NEAT Server',
    logs: getAllLogs(),
    clientCount: wss.clients.size,
    timestamp: new Date().toISOString()
  });
});

// WebSocket connection handling
wss.on('connection', (ws: WebSocket, request) => {
  const clientIP = request.socket.remoteAddress || 'unknown';
  console.log(`🔗 New WebSocket connection from ${clientIP}`);

  // Send welcome message
  ws.send(JSON.stringify({
    type: 'welcome',
    message: 'Connected to NEAT Server WebSocket',
    timestamp: new Date().toISOString(),
    clientId: generateClientId()
  }));

  // Handle incoming messages
  ws.on('message', (data: Buffer) => {
    try {
      const message = JSON.parse(data.toString());
      console.log('📨 Received:', message);

      // Handle different message types
      switch (message.type) {
        case 'error':
          handleClientError(message, clientIP);
          break;
        case 'warning':
          handleClientWarning(message, clientIP);
          break;
        case 'info':
          handleClientInfo(message, clientIP);
          break;
        case 'performance':
          handlePerformanceData(message, clientIP);
          break;
        case 'user_action':
          handleUserAction(message, clientIP);
          break;
        default:
          // Regular message - echo back
          ws.send(JSON.stringify({
            type: 'echo',
            original: message,
            server: 'NEAT Server',
            timestamp: new Date().toISOString()
          }));

          // Broadcast to all other clients
          broadcastToOthers(ws, {
            type: 'broadcast',
            from: 'client',
            message: message,
            timestamp: new Date().toISOString()
          });
          break;
      }

    } catch (error) {
      console.error('❌ Error parsing message:', error);
      ws.send(JSON.stringify({
        type: 'error',
        message: 'Invalid JSON format',
        timestamp: new Date().toISOString()
      }));
    }
  });

  // Handle connection close
  ws.on('close', (code, reason) => {
    console.log(`🔌 WebSocket disconnected: ${code} - ${reason}`);
  });

  // Handle errors
  ws.on('error', (error) => {
    console.error('❌ WebSocket error:', error);
  });

  // Send periodic ping to keep connection alive
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'ping',
        timestamp: new Date().toISOString()
      }));
    } else {
      clearInterval(pingInterval);
    }
  }, 30000); // Every 30 seconds
});

// Helper functions
function generateClientId(): string {
  return Math.random().toString(36).substr(2, 9);
}

function broadcastToOthers(sender: WebSocket, message: any): void {
  wss.clients.forEach((client) => {
    if (client !== sender && client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(message));
    }
  });
}

// Client error handling functions
function handleClientError(message: any, clientIP: string): void {
  const errorInfo = {
    timestamp: new Date().toISOString(),
    clientIP: clientIP,
    type: 'CLIENT_ERROR',
    error: message.error,
    stack: message.stack,
    url: message.url,
    line: message.line,
    column: message.column,
    userAgent: message.userAgent
  };

  console.error('🚨 CLIENT ERROR:', errorInfo);
  
  // Log to file or external service here if needed
  logClientError(errorInfo);
  
  // Broadcast error to monitoring clients
  broadcastToMonitoring({
    type: 'client_error',
    data: errorInfo
  });
}

function handleClientWarning(message: any, clientIP: string): void {
  const warningInfo = {
    timestamp: new Date().toISOString(),
    clientIP: clientIP,
    type: 'CLIENT_WARNING',
    warning: message.warning,
    context: message.context
  };

  console.warn('⚠️ CLIENT WARNING:', warningInfo);
  logClientWarning(warningInfo);
}

function handleClientInfo(message: any, clientIP: string): void {
  const infoData = {
    timestamp: new Date().toISOString(),
    clientIP: clientIP,
    type: 'CLIENT_INFO',
    info: message.info,
    details: message.details
  };

  console.info('ℹ️ CLIENT INFO:', infoData);
  logClientInfo(infoData);
}

function handlePerformanceData(message: any, clientIP: string): void {
  const perfData = {
    timestamp: new Date().toISOString(),
    clientIP: clientIP,
    type: 'PERFORMANCE',
    metrics: message.metrics
  };

  console.log('📊 PERFORMANCE DATA:', perfData);
  logPerformanceData(perfData);
}

function handleUserAction(message: any, clientIP: string): void {
  const actionData = {
    timestamp: new Date().toISOString(),
    clientIP: clientIP,
    type: 'USER_ACTION',
    action: message.action,
    element: message.element,
    data: message.data
  };

  console.log('👤 USER ACTION:', actionData);
  logUserAction(actionData);
}

// Logging functions
function logClientError(errorInfo: any): void {
  // Store in memory
  logs.errors.push(errorInfo);
  // Keep only last 100 errors
  if (logs.errors.length > 100) {
    logs.errors = logs.errors.slice(-100);
  }
  
  const logEntry = `[${errorInfo.timestamp}] ERROR from ${errorInfo.clientIP}: ${errorInfo.error}\n`;
  console.error(logEntry);
}

function logClientWarning(warningInfo: any): void {
  logs.warnings.push(warningInfo);
  if (logs.warnings.length > 100) {
    logs.warnings = logs.warnings.slice(-100);
  }
  
  const logEntry = `[${warningInfo.timestamp}] WARNING from ${warningInfo.clientIP}: ${warningInfo.warning}\n`;
  console.warn(logEntry);
}

function logClientInfo(infoData: any): void {
  logs.info.push(infoData);
  if (logs.info.length > 100) {
    logs.info = logs.info.slice(-100);
  }
  
  const logEntry = `[${infoData.timestamp}] INFO from ${infoData.clientIP}: ${infoData.info}\n`;
  console.info(logEntry);
}

function logPerformanceData(perfData: any): void {
  logs.performance.push(perfData);
  if (logs.performance.length > 50) {
    logs.performance = logs.performance.slice(-50);
  }
  
  const logEntry = `[${perfData.timestamp}] PERF from ${perfData.clientIP}: ${JSON.stringify(perfData.metrics)}\n`;
  console.log(logEntry);
}

function logUserAction(actionData: any): void {
  logs.userActions.push(actionData);
  if (logs.userActions.length > 200) {
    logs.userActions = logs.userActions.slice(-200);
  }
  
  const logEntry = `[${actionData.timestamp}] ACTION from ${actionData.clientIP}: ${actionData.action}\n`;
  console.log(logEntry);
}

function getAllLogs() {
  return {
    summary: {
      totalErrors: logs.errors.length,
      totalWarnings: logs.warnings.length,
      totalInfo: logs.info.length,
      totalPerformance: logs.performance.length,
      totalUserActions: logs.userActions.length
    },
    recent: {
      errors: logs.errors.slice(-10),
      warnings: logs.warnings.slice(-10),
      info: logs.info.slice(-10),
      performance: logs.performance.slice(-5),
      userActions: logs.userActions.slice(-20)
    },
    all: logs
  };
}

function broadcastToMonitoring(data: any): void {
  // Broadcast to clients that are listening for monitoring data
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({
        type: 'monitoring',
        data: data,
        timestamp: new Date().toISOString()
      }));
    }
  });
}

// Server startup
const PORT = process.env.PORT || 3000;

server.listen(PORT, () => {
  console.log('🚀 NEAT Server starting...');
  console.log(`📡 HTTP Server: http://localhost:${PORT}`);
  console.log(`🔌 WebSocket: ws://localhost:${PORT}`);
  console.log(`👥 Connected clients: ${wss.clients.size}`);
  console.log('✅ Server is ready!');
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('🛑 Received SIGTERM, shutting down gracefully...');
  server.close(() => {
    console.log('✅ Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  console.log('🛑 Received SIGINT, shutting down gracefully...');
  server.close(() => {
    console.log('✅ Server closed');
    process.exit(0);
  });
});

export default app;