<?php
// password to login to phpLiteAdmin
// if you change this, you must delete the cookie from your browser
$password = 'nogiyu'; // 例: 'MySecretPa$$w0rd'

// directory of databases
// optional, if you want to manage databases in a different directory
$directory = '.'; // カレントディレクトリ (phpliteadmin.phpと同じ場所) のDBを対象とする

// an array of databases that you want to manage
// or false to manage all databases in the directory
$databases = array(
    array(
        'path'=> 'annotation_data.db', // 管理したいDBファイル名
        'name'=> 'Annotation DB'    // phpLiteAdmin上での表示名
    ),
);

// For more configuration options, see phpliteadmin.php source code
?>
