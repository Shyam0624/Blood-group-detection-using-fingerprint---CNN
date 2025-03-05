<?php
header("Content-Type: application/json");

try {
    if ($_FILES['image']['error'] !== UPLOAD_ERR_OK) {
        throw new Exception("File upload error.");
    }

    // Load the uploaded image into memory
    $tmpFilePath = $_FILES['image']['tmp_name'];

    // Call Python script to predict the blood group
    $command = escapeshellcmd("python C:/xampp/htdocs/mini/process_image.py") . ' ' . escapeshellarg($tmpFilePath) . " 2>&1";

    $output = shell_exec($command);

    // Log debug output (optional)
    file_put_contents('debug_log.txt', $output);

    if ($output === null || strpos($output, "Error:") !== false) {
        throw new Exception("Error executing the prediction model: $output");
    }

    // Clean the output
    $prediction = trim($output);

    echo json_encode([
        "success" => true,
        "prediction" => $prediction
    ]);
} catch (Exception $e) {
    file_put_contents('debug_log.txt', $e->getMessage(), FILE_APPEND); // Log exception details
    echo json_encode([
        "success" => false,
        "message" => $e->getMessage()
    ]);
}

?>
