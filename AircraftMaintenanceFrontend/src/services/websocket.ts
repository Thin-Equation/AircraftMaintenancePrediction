import { TrainingNotification } from '@/types';

type NotificationCallback = (notification: TrainingNotification) => void;

/**
 * WebSocket service for receiving training notifications from the backend
 */
class WebSocketService {
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimeout = 3000; // Start with 3 seconds
  private callbacks: NotificationCallback[] = [];
  private isConnecting = false;

  /**
   * Connect to the WebSocket server
   */
  connect(): Promise<void> {
    if (this.socket?.readyState === WebSocket.OPEN || this.isConnecting) {
      return Promise.resolve();
    }

    this.isConnecting = true;
    
    return new Promise((resolve, reject) => {
      const wsUrl = `${this.getWebSocketBaseUrl()}/ws/training`;
      this.socket = new WebSocket(wsUrl);

      this.socket.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.isConnecting = false;
        resolve();
      };

      this.socket.onclose = (event) => {
        this.isConnecting = false;
        if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
          console.log(`WebSocket connection closed. Reconnecting attempt ${this.reconnectAttempts + 1}...`);
          this.reconnectWithBackoff();
        } else {
          console.log('WebSocket connection closed');
        }
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.isConnecting = false;
        reject(error);
      };

      this.socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as TrainingNotification;
          if (data.type === 'training_update') {
            this.notifyCallbacks(data);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
    });
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }

  /**
   * Add a callback to be notified when a training update is received
   */
  onTrainingUpdate(callback: NotificationCallback): void {
    this.callbacks.push(callback);
  }

  /**
   * Remove a callback
   */
  removeCallback(callback: NotificationCallback): void {
    this.callbacks = this.callbacks.filter(cb => cb !== callback);
  }

  /**
   * Send a message through the WebSocket
   */
  sendMessage(message: string): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(message);
    } else {
      console.error('WebSocket is not connected');
    }
  }

  /**
   * Get the WebSocket base URL based on the current environment
   */
  private getWebSocketBaseUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = process.env.NEXT_PUBLIC_API_URL 
      ? new URL(process.env.NEXT_PUBLIC_API_URL).host
      : 'localhost:8000';
      
    return `${protocol}//${host}`;
  }

  /**
   * Reconnect with exponential backoff
   */
  private reconnectWithBackoff(): void {
    this.reconnectAttempts++;
    const backoffTime = this.reconnectTimeout * Math.pow(2, this.reconnectAttempts - 1);
    
    setTimeout(() => {
      this.connect().catch(() => {
        // Error handling is already in the connect method
      });
    }, backoffTime);
  }

  /**
   * Notify all registered callbacks of a training update
   */
  private notifyCallbacks(notification: TrainingNotification): void {
    this.callbacks.forEach(callback => {
      try {
        callback(notification);
      } catch (error) {
        console.error('Error in notification callback:', error);
      }
    });
  }
}

// Export a singleton instance
export const wsService = new WebSocketService();
export default wsService;