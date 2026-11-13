[CmdletBinding()]
param(
    [string]$SdkPath,
    [string]$JdkPath,
    [string]$GradleHome,
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$androidProject = Join-Path $repositoryRoot 'clients\glasses\android'
$originalEnvironment = @{}
foreach ($name in @('PATH', 'JAVA_HOME', 'ANDROID_HOME', 'ANDROID_SDK_ROOT')) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable exited with code $LASTEXITCODE" }
}

try {
    # Reuse an installed Unity toolchain when no explicit SDK/JDK was supplied.
    $unityAndroid = $null
    $unityEditors = Join-Path $env:ProgramFiles 'Unity\Hub\Editor'
    if (Test-Path -LiteralPath $unityEditors) {
        foreach ($editor in (Get-ChildItem -LiteralPath $unityEditors -Directory | Sort-Object Name -Descending)) {
            $candidate = Join-Path $editor.FullName 'Editor\Data\PlaybackEngines\AndroidPlayer'
            if (Test-Path -LiteralPath (Join-Path $candidate 'SDK\platforms\android-36\android.jar')) {
                $unityAndroid = $candidate
                break
            }
        }
    }
    if (-not $SdkPath) {
        if ($env:ANDROID_HOME) { $SdkPath = $env:ANDROID_HOME }
        elseif ($env:ANDROID_SDK_ROOT) { $SdkPath = $env:ANDROID_SDK_ROOT }
        elseif ($unityAndroid) { $SdkPath = Join-Path $unityAndroid 'SDK' }
        else { $SdkPath = Join-Path $env:LOCALAPPDATA 'Android\Sdk' }
    }
    if (-not $JdkPath) {
        if ($env:JAVA_HOME) { $JdkPath = $env:JAVA_HOME }
        elseif ($unityAndroid) { $JdkPath = Join-Path $unityAndroid 'OpenJDK' }
        else { throw 'Pass -JdkPath pointing to JDK 17 or newer (or set JAVA_HOME).' }
    }
    $SdkPath = [IO.Path]::GetFullPath($SdkPath)
    $JdkPath = [IO.Path]::GetFullPath($JdkPath)
    $java = Join-Path $JdkPath 'bin\java.exe'
    $buildTools = Join-Path $SdkPath 'build-tools\36.0.0'
    foreach ($required in @($java, (Join-Path $SdkPath 'platforms\android-36\android.jar'),
                            (Join-Path $buildTools 'apksigner.bat'), (Join-Path $buildTools 'aapt2.exe'))) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing build prerequisite: $required" }
    }
    $env:JAVA_HOME = $JdkPath
    $env:ANDROID_HOME = $SdkPath
    $env:ANDROID_SDK_ROOT = $SdkPath
    $env:PATH = (Join-Path $JdkPath 'bin') + ';' + $env:PATH

    if (-not $Python) {
        $localPython = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
        $Python = if (Test-Path -LiteralPath $localPython) { $localPython } else { (Get-Command python -ErrorAction Stop).Source }
    }
    $Python = (Get-Command $Python -ErrorAction Stop).Source
    # Conda-derived venvs need their base DLL directory, without changing the user's installation.
    $venvConfig = Join-Path (Split-Path (Split-Path $Python)) 'pyvenv.cfg'
    if (Test-Path -LiteralPath $venvConfig) {
        $baseLine = Get-Content -LiteralPath $venvConfig | Where-Object { $_ -match '^home\s*=\s*(.+)$' } | Select-Object -First 1
        if ($baseLine -match '^home\s*=\s*(.+)$') {
            $pythonBase = $Matches[1]
            $condaDlls = Join-Path $pythonBase 'Library\bin'
            if (Test-Path -LiteralPath $condaDlls) { $env:PATH = $condaDlls + ';' + $pythonBase + ';' + $env:PATH }
        }
    }
    Invoke-Checked $Python @((Join-Path $repositoryRoot 'tools\prepare_assist_assets.py'))

    if (-not $GradleHome -and $unityAndroid) {
        $unityGradle = Join-Path $unityAndroid 'Tools\gradle'
        if (Test-Path -LiteralPath (Join-Path $unityGradle 'lib\gradle-launcher-9.1.0.jar')) { $GradleHome = $unityGradle }
    }
    $gradleArgs = @('--project-dir', $androidProject, '--console', 'plain', '--no-daemon',
                    "-PassistPython=$Python", 'testDebugUnitTest', 'lintDebug', 'assembleDebug')
    if ($GradleHome) {
        $launcher = Join-Path $GradleHome 'lib\gradle-launcher-9.1.0.jar'
        if (-not (Test-Path -LiteralPath $launcher)) { throw 'GradleHome must contain Gradle 9.1.0.' }
        Invoke-Checked $java (@('-classpath', $launcher, 'org.gradle.launcher.GradleMain') + $gradleArgs)
    } else {
        Invoke-Checked (Join-Path $androidProject 'gradlew.bat') $gradleArgs
    }

    $builtApk = Join-Path $androidProject 'app\build\outputs\apk\debug\app-debug.apk'
    Invoke-Checked $Python @((Join-Path $repositoryRoot 'tools\prepare_assist_assets.py'), '--verify-apk', $builtApk)
    $signerOutput = & (Join-Path $buildTools 'apksigner.bat') verify --verbose --print-certs $builtApk
    if ($LASTEXITCODE -ne 0) { throw 'APK signature verification failed.' }
    $badging = & (Join-Path $buildTools 'aapt2.exe') dump badging $builtApk
    if ($LASTEXITCODE -ne 0) { throw 'APK manifest inspection failed.' }
    if (-not ($badging -match "package: name='org\.cvision\.assist' versionCode='2' versionName='0\.2\.0'")) {
        throw 'Unexpected package identity/version in the built APK.'
    }
    if (-not ($badging -match "^(minSdkVersion|sdkVersion):'26'$") -or -not ($badging -contains "targetSdkVersion:'36'")) {
        throw 'Unexpected min/target SDK in the built APK.'
    }
    $dist = Join-Path $repositoryRoot 'dist'
    New-Item -ItemType Directory -Path $dist -Force | Out-Null
    $apk = Join-Path $dist 'C-Vision-Assist-0.2.0-debug.apk'
    Copy-Item -LiteralPath $builtApk -Destination $apk -Force
    $digest = (Get-FileHash -LiteralPath $apk -Algorithm SHA256).Hash.ToLowerInvariant()
    [IO.File]::WriteAllText($apk + '.sha256', "$digest  $([IO.Path]::GetFileName($apk))`n", [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllLines($apk + '.signer.txt', $signerOutput, [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllLines($apk + '.manifest.txt', $badging, [Text.UTF8Encoding]::new($false))
    Write-Output $signerOutput
    Write-Output "Built and verified: $apk"
    Write-Output "SHA-256: $digest"
} finally {
    foreach ($name in $originalEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $originalEnvironment[$name], 'Process')
    }
}
