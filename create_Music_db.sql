-- @block
CREATE TABLE Users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50)
);
-- @block
CREATE TABLE Songs (
    song_id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(200),
    artist VARCHAR(800),
    emotion VARCHAR(20),
    album VARCHAR(255),
    release_date DATE,
    energy FLOAT,
    danceability FLOAT,
    positiveness FLOAT,
    speechiness FLOAT,
    liveness FLOAT,
    acousticness FLOAT,
    instrumentalness FLOAT
);
-- @block
CREATE TABLE Genres (
    genre_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) UNIQUE
);

CREATE TABLE Song_Genres (
    song_id INT,
    genre_id INT,
    PRIMARY KEY (song_id, genre_id),
    FOREIGN KEY (song_id) REFERENCES Songs(song_id),
    FOREIGN KEY (genre_id) REFERENCES Genres(genre_id)
);
-- @block
CREATE TABLE Personality_Types (
    personality_type VARCHAR(20) PRIMARY KEY,
    description TEXT,
    energy_level VARCHAR(10),
    danceability_level VARCHAR(10),
    positiveness_level VARCHAR(10),
    speechiness_level VARCHAR(10),
    liveness_level VARCHAR(10),
    acousticness_level VARCHAR(10),
    instrumentalness_level VARCHAR(10)
);
-- @block
CREATE TABLE User_Selected_Songs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    song_id INT,
    selected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY user_song_unique (user_id, song_id),
    FOREIGN KEY (user_id) REFERENCES Users(user_id),
    FOREIGN KEY (song_id) REFERENCES Songs(song_id)
);
-- @block
CREATE TABLE Personality_Result (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    personality_type VARCHAR(20),
    avg_energy FLOAT,
    avg_danceability FLOAT,
    avg_speechiness FLOAT,
    avg_acousticness FLOAT,
    FOREIGN KEY (user_id) REFERENCES Users(user_id),
    FOREIGN KEY (personality_type) REFERENCES Personality_Types(personality_type)

);
-- @block
CREATE TABLE Recommended_Songs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    personality_type VARCHAR(20),
    song_id INT,
    FOREIGN KEY (song_id) REFERENCES Songs(song_id),
    FOREIGN KEY (personality_type) REFERENCES Personality_Types(personality_type)
);
