pipeline {
  agent any

  environment {
    IMAGE = "karledroosvelt/pro2"
    PORT  = "8501"
  }

  stages {

    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Install & Tests') {
      steps {
        bat '''
          python --version
          pip install -U pip
          pip install -r requirements.txt
          pip install pytest
          pytest -q
        '''
      }
    }

    stage('Build Docker Image') {
      steps {
        bat '''
          for /f %%i in ('git rev-parse --short HEAD') do set COMMIT=%%i
          echo %COMMIT% > .git_commit
          docker build -t %IMAGE%:%COMMIT% -t %IMAGE%:latest .
        '''
      }
    }

    stage('DockerHub Login') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'dockerhub-creds', usernameVariable: 'DH_USER', passwordVariable: 'DH_PASS')]) {
          bat '''
            echo %DH_PASS% | docker login -u %DH_USER% --password-stdin
          '''
        }
      }
    }

    stage('Push Image') {
      steps {
        bat '''
          set /p COMMIT=<.git_commit
          docker push %IMAGE%:%COMMIT%
          docker push %IMAGE%:latest
        '''
      }
    }

    stage('Deploy Automatically') {
      when { branch "main" }
      steps {
        bat '''
          docker rm -f pro2-streamlit 2>nul
          docker pull %IMAGE%:latest
          docker run -d --name pro2-streamlit -p %PORT%:%PORT% %IMAGE%:latest
        '''
      }
    }
  }
}
