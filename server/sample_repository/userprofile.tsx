// UserProfile.tsx
import React, { useState, useEffect } from 'react';
import { formatDate, calculateAge } from './utils';
import { ApiClient } from './api';

const client = new ApiClient("https://api.dummy.com");

export const UserProfile = ({ userId }: { userId: string }) => {
    /** Main React Component for displaying a user's profile. */
    
    // React Hooks (Standalone calls)
    const [user, setUser] = useState<any>(null);

    useEffect(() => {
        // Nested arrow function inside a hook
        const loadUser = async () => {
            // Object method call from the imported class
            const data = await client.fetchUserData(userId);
            setUser(data);
        };
        
        loadUser();
    }, [userId]);

    if (!user) {
        return <div>Loading...</div>;
    }

    return (
        <div className="profile-card">
            <h1>{user.name}</h1>
            {/* Standalone calls imported from utils */}
            <p>Age: {calculateAge(user.birthYear)}</p>
            <p>Joined: {formatDate(user.joinDate)}</p>
            
            <div className="hobbies">
                <h3>Hobbies</h3>
                <ul>
                    {/* Method call (.map) on an array inside JSX */}
                    {user.hobbies.map((hobby: string) => (
                        <li key={hobby}>{hobby}</li>
                    ))}
                </ul>
            </div>
        </div>
    );
};