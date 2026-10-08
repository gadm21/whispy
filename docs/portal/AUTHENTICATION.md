# Authentication Flow

This document outlines the authentication flow in the thothHUB application.

## Overview

The authentication system uses JWT (JSON Web Tokens) for securing API endpoints and managing user sessions. The flow is as follows:

1. User submits login credentials
2. Backend validates credentials and returns a JWT
3. The token is stored in localStorage and managed by the AuthContext
4. All subsequent API requests include the token in the Authorization header
5. The token is validated on protected routes and API endpoints

## Key Components

### AuthContext (`/contexts/AuthContext.tsx`)

- Manages the authentication state
- Provides login/logout functionality
- Handles token storage and retrieval
- Provides user information to the application

### Protected Routes (`/app/(protected)/layout.tsx`)

- Wraps all protected routes
- Redirects unauthenticated users to the login page
- Shows a loading state while checking authentication

### API Client (`/hooks/useApi.ts`)

- Handles authenticated API requests
- Automatically includes the JWT in request headers
- Handles token expiration and 401 responses
- Provides a clean interface for making HTTP requests

## Usage

### Using the Auth Context

```tsx
import { useAuth } from '@/contexts/AuthContext';

function MyComponent() {
  const { user, isAuthenticated, login, logout } = useAuth();
  
  // Access user info
  console.log(user?.username);
  
  // Check authentication status
  if (!isAuthenticated) {
    return <div>Please log in</div>;
  }
  
  return (
    <div>
      Welcome, {user?.username}!
      <button onClick={logout}>Logout</button>
    </div>
  );
}
```

### Making Authenticated API Requests

```tsx
import { useApi } from '@/hooks/useApi';

function MyComponent() {
  const { get, post, put, delete: del } = useApi();
  
  const fetchData = async () => {
    try {
      // GET request
      const devices = await get('/devices');
      
      // POST request
      await post('/devices', { name: 'New Device' });
      
      // PUT request
      await put('/devices/123', { name: 'Updated Device' });
      
      // DELETE request
      await del('/devices/123');
      
    } catch (error) {
      console.error('API Error:', error);
    }
  };
  
  // ...
}
```

## Error Handling

- Authentication errors (401) automatically log the user out and redirect to login
- API errors are thrown as exceptions with descriptive messages
- The `useApi` hook handles token refresh and retry logic

## Security Considerations

- JWT is stored in localStorage (consider httpOnly cookies for enhanced security)
- Tokens have an expiration time (handled by the backend)
- All API requests include CSRF protection via the `credentials: 'include'` option
- Sensitive routes are protected on both client and server side
