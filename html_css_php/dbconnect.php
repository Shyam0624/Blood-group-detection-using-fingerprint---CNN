<?php
// Database connection file: dbconnect.php

$host = 'localhost';       // Hostname
$username = 'root';        // Default username for localhost
$password = '';            // Default password for localhost
$dbname = 'project2';      // Your database name

// Create a connection
$conn = new mysqli($host, $username, $password, $dbname);

// Check the connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Connection successful message (for debugging - optional)
// echo "Connected successfully";
?>
