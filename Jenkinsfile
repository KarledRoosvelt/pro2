pipeline {
  agent any

  environment {
    IMAGE = "karledroosvelt/pro2"
    PORT  = "8501"
    EMAIL_TO = "roosveltkenfi@gmail.com"
  }

  stages {

    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Install & Tests') {
      steps {
        sh '''
          pip install -U pip
          pip install -r requirements.txt
          pip install pytest
          pytest -q
        '''
      }
    }

    stage('Build Docker Image') {
      steps {
        sh '''
          COMMIT=$(git rev-parse --short HEAD)
          echo $COMMIT > .git_commit
          docker build -t $IMAGE:$COMMIT -t $IMAGE:latest .
        '''
      }
    }

    stage('DockerHub Login') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'dockerhub-creds', usernameVariable: 'DH_USER', passwordVariable: 'DH_PASS')]) {
          sh 'echo $DH_PASS | docker login -u $DH_USER --password-stdin'
        }
      }
    }

    stage('Push Image') {
      steps {
        sh '''
          COMMIT=$(cat .git_commit)
          docker push $IMAGE:$COMMIT
          docker push $IMAGE:latest
        '''
      }
    }

    stage('Deploy Automatically') {
      when { branch "main" }
      steps {
        sh '''
          docker rm -f pro2-streamlit || true
          docker pull $IMAGE:latest
          docker run -d --name pro2-streamlit -p $PORT:$PORT $IMAGE:latest
        '''
      }
    }
  }

  post {
    success {
      emailext (
        subject: "✅ Jenkins SUCCESS : ${JOB_NAME} #${BUILD_NUMBER}",
        body: "Le déploiement a réussi.\n\nJob: ${JOB_NAME}\nBuild: ${BUILD_NUMBER}",
        to: "${EMAIL_TO}"
      )
    }

    failure {
      emailext (
        subject: "❌ Jenkins FAILURE : ${JOB_NAME} #${BUILD_NUMBER}",
        body: """Le pipeline a échoué.

Job: ${JOB_NAME}
Build: ${BUILD_NUMBER}
URL: ${BUILD_URL}

Merci de consulter les logs Jenkins.
""",
        to: "${EMAIL_TO}"
      )
    }
  }
}
