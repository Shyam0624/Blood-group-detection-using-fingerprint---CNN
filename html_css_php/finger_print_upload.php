<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="styless.css">
    <title>BGD | Home</title>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
</head>
<body>
    <div class="wrapper">
        <nav class="nav">
            <div class="nav-logo">
                <p>BLOODGROUP DETECTOR</p>
            </div>
            <div class="nav-menu" id="navMenu">
                <ul>
                    <li><a href="#" class="link active">Home</a></li>
                    <li><a href="about.html" class="link">About</a></li>
                </ul>
            </div>

            <div class="nav-button" id="navMenu">
                <a href="index.php"><button class="btn white-btn" id="loginBtn">Log Out</button></a>
            </div>
            <div class="nav-menu-btn">
                <i class="bx bx-menu" onclick="myMenuFunction()"></i>
            </div>
        </nav>
        

        <section class="hero">
            <div class="hero-content">
                <h1>Upload <span class="highlight">fingerprint</span> to Check</h1>
                <p>New way of Detecting Blood Group with Fingerprint</p>
                <div class="hero-buttons">
                    <button class="btn hero-btn">Get Started</button>
                    <button class="btn hero-btn"><a href="https://youtu.be/-aWnexbnVOw?si=buGVIZ8Q0xoZokOo" class="text-color">Watch Video</a></button>
                </div>
            </div>
            <div class="hero-content">
                <div>
                    <br>
                    <h1>Blood Group Prediction</h1>
                </div>
                <div class="hero-image">
                    <img src="upload.jpg" alt="Lab Illustration">
                </div>
                <div class="hero-buttons">
                    <br>
                    <br>
                    <form id="uploadForm">
                        <label for="image">Upload an image:  </label>
                        <input type="file" id="image" name="image" accept="image/*" required>
                        <br>
                        <br>
                        <button type="submit" class="btn hero-btn">Predict</button>
                    </form>
                    <div id="result"></div>
                </div>
            </div>
            

            
        </section>
    </div>

    <script>
        function myMenuFunction() {
            var i = document.getElementById("navMenu");
            if (i.className === "nav-menu") {
                i.className += " responsive";
            } else {
                i.className = "nav-menu";
            }
        }

        $(document).ready(function () {
            $('#uploadForm').submit(function (e) {
                e.preventDefault();

                var formData = new FormData();
                var fileInput = $('#image')[0].files[0];

                if (!fileInput) {
                    $('#result').text("Please select an image.");
                    return;
                }

                formData.append('image', fileInput);

                // Show loading message
                $('#result').text("Processing...");

                // AJAX call to `upload.php`
                $.ajax({
                    url: 'upload.php',
                    type: 'POST',
                    data: formData,
                    processData: false,
                    contentType: false,
                    success: function (response) {
                        if (response.success) {
                            // Extract only the prediction (e.g., AB-) using a regex
                            const cleanPrediction = response.prediction.match(/(AB|A|B|O)[+-]/)?.[0] || "Unknown";
        
                            // Display the clean prediction
                            $('#result').html(`<p>Prediction: <strong>${cleanPrediction}</strong></p>`);
                        } else {
                            $('#result').html(`<p>Error: ${response.message}</p>`);
                        }
                    },
                    error: function () {
                        $('#result').html("<p>An error occurred while processing the image.</p>");
                    }
                });
            });
        });
    </script>
</body>
</html>
