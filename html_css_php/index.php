<?php
// Include the database connection file
include 'dbconnect.php';

// Handle Registration
if (isset($_POST['register'])) {
    $firstname = $_POST['firstname'];
    $lastname = $_POST['lastname'];
    $email = $_POST['email'];
    $password = $_POST['password']; // Encrypt the password

    // Insert into users table
    $sql = "INSERT INTO users (firstname, lastname, email, password) VALUES ('$firstname', '$lastname', '$email', '$password')";
    if ($conn->query($sql) === TRUE) {
        echo "<script>alert('Registration successful! Please login.');</script>";
    } else {
        echo "<script>alert('Error: " . $conn->error . "');</script>";
    }
}

// Handle Login
if (isset($_POST['login'])) {
    $email = $_POST['email'];
    $password = $_POST['password'];

    // Check user in the database
    $sql = "SELECT * FROM users WHERE email = '$email'";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        $row = $result->fetch_assoc();
        // Verify the password
        if ($password == $row['password']) {
            echo "<script>alert('Login successful. Welcome, " . $row['firstname'] . "!');</script>";
            // Redirect to another page (optional)
            header('Location: finger_print_upload.php');
        } else {

            echo "<script>alert('Invalid password');</script>";
        }
    } else {
        echo "<script>alert('No user found with this email.');</script>";
    }
}

$conn->close();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="styless.css">
    <title>BGD | Sign In & Sign Up</title>
</head>
<body>
    <nav class="nav">
        <div class="nav-logo">
            <p>BLOODGROUP DETECTOR</p>
        </div>
        <div class="nav-menu">
            <ul>
                <li><a href="#" class="link active">Sign In/Sign Up</a></li>
            </ul>
        </div>
    </nav>

    <div class="wrapper">
        <div class="form-box">
            <!-- Login Form -->
            <div class="login-container" id="login">
                <div class="top">
                    <span>Don't have an account? <a href="#" onclick="register()">Sign Up</a></span>
                    <header>Login</header>
                </div>

                <form method="POST" action="index.php">
                    <div class="input-box">
                        <input type="text" name="email" class="input-field" placeholder="Username or Email" required>
                    </div>
                    <br>
                    <div class="input-box">
                        <input type="password" name="password" class="input-field" placeholder="Password" required>
                    </div>
                    <br>
                    <br>
                    <div class="input-box">
                        <input type="submit" name="login" class="submit" value="Sign In">
                    </div>
                </form>

                <div class="two-col">
                    <div class="one">
                        <input type="checkbox" id="login-check">
                        <label for="login-check"> Remember Me</label>
                    </div>
                    <div class="two">
                        <label><a href="#">Forgot password?</a></label>
                    </div>
                </div>
            </div>

            <!-- Registration Form -->
            <div class="register-container" id="register">
                <div class="top">
                    <span>Have an account? <a href="#" onclick="login()">Login</a></span>
                    <header>Sign Up</header>
                </div>

                <form method="POST" action="index.php">
                    <div class="two-forms">
                        <div class="input-box">
                            <input type="text" name="firstname" class="input-field" placeholder="Firstname" required>
                        </div>
                        <div class="input-box">
                            <input type="text" name="lastname" class="input-field" placeholder="Lastname" required>
                        </div>
                    </div>
                    <br>
                    <div class="input-box">
                        <input type="text" name="email" class="input-field" placeholder="Email" required>
                    </div>
                    <br>
                    <div class="input-box">
                        <input type="password" name="password" class="input-field" placeholder="Password" required>
                    </div>
                    <br>
                    <div class="input-box">
                        <input type="submit" name="register" class="submit" value="Register">
                    </div>
                </form>

                <div class="two-col">
                    <div class="one">
                        <input type="checkbox" id="register-check">
                        <label for="register-check"> Remember Me</label>
                    </div>
                    <div class="two">
                        <label><a href="#">Terms & conditions</a></label>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        var x = document.getElementById("login");
        var y = document.getElementById("register");

        function login() {
            x.style.left = "4px";
            y.style.right = "-520px";
            x.style.opacity = 1;
            y.style.opacity = 0;
        }

        function register() {
            x.style.left = "-510px";
            y.style.right = "5px";
            x.style.opacity = 0;
            y.style.opacity = 1;
        }
    </script>
</body>
</html>
