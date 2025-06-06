import sqlite3
from datetime import datetime
import os

class Memory:
    def __init__(self, db_file="memory.db"):
        self.db_file = db_file
        self.conn = None
        self.cursor = None
        self.init_db()

    def init_db(self):
        """Initialize the database"""
        try:
            self.conn = sqlite3.connect(self.db_file)
            self.cursor = self.conn.cursor()
            
            # Create table if it doesn't exist
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            ''')
            
            # Create index for faster queries
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON conversation_history(timestamp)
            ''')
            
            self.conn.commit()
        except Exception as e:
            print(f"Error initializing database: {e}")

    def add_to_history(self, role, content):
        """Add a message to the conversation history"""
        try:
            self.cursor.execute('''
                INSERT INTO conversation_history (role, content, timestamp)
                VALUES (?, ?, ?)
            ''', (role, content, datetime.now().isoformat()))
            
            # Keep only the last 1000 messages
            self.cursor.execute('''
                DELETE FROM conversation_history
                WHERE id NOT IN (
                    SELECT id FROM conversation_history
                    ORDER BY timestamp DESC
                    LIMIT 1000
                )
            ''')
            
            self.conn.commit()
        except Exception as e:
            print(f"Error adding to history: {e}")

    def get_recent_history(self, max_messages=10):
        """Get recent conversation history"""
        try:
            self.cursor.execute('''
                SELECT role, content
                FROM conversation_history
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (max_messages,))
            
            # Convert to the format OpenAI expects
            messages = []
            for role, content in reversed(self.cursor.fetchall()):
                messages.append({"role": role, "content": content})
            return messages
        except Exception as e:
            print(f"Error getting history: {e}")
            return []

    def clear_memory(self):
        """Clear the conversation history"""
        try:
            self.cursor.execute('DELETE FROM conversation_history')
            self.conn.commit()
        except Exception as e:
            print(f"Error clearing memory: {e}")

    def __del__(self):
        """Clean up database connection"""
        if self.conn:
            self.conn.close() 