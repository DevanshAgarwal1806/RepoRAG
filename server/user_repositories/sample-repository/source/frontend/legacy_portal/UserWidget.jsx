import React from 'react';

/**
 * Renders a simple card displaying user avatar and handle.
 */
const UserWidget = ({ username, avatarUrl }) => {
    return (
        <div className="widget-card">
            <img src={avatarUrl} alt={`${username}'s avatar`} />
            <h3>@{username}</h3>
        </div>
    );
};

export default UserWidget;
