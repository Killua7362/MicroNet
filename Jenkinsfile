pipeline {
    environment {
        PROJECT = "clowder-403113"
        APP_NAME = "test-app"
        REPO_NAME = "testing"
        REPO_LOCATION = "asia-south1"
        IMAGE_NAME = "${REPO_LOCATION}-docker.pkg.dev/${PROJECT}/${REPO_NAME}/${APP_NAME}"
        CRED_ID = 'gcloud-creds'
    }
    agent any
    stages {
        stage('Pull Git') {
            steps {
                echo 'Cloning..'
                deleteDir() // Clean the workspace
                checkout scm
                }
        }
        stage('Build and push docker image'){
                steps{
                    script{
                    sh 'docker build -t ${IMAGE_NAME}:lastest -f Dockerfile .'
                    withCredentials([file(credentialsId: "${CRED_ID}", variable: 'GCR_CRED')]){
                        sh 'cat "${GCR_CRED}" | docker login -u _json_key --password-stdin https://"${REPO_LOCATION}"-docker.pkg.dev'
                        sh 'docker push ${IMAGE_NAME}:lastest'
                        sh 'docker logout https://${REPO_LOCATION}-docker.pkg.dev'
                    }                
                        sh 'docker rmi ${IMAGE_NAME}:lastest'
                    }
                }
            }
        stage('Upload to gke'){
            steps{
                script{
                    sh "kubectl apply -f kubernetes/backend.yaml"
                    sh "kubectl apply -f kubernetes/service.yaml"
                }
            }
        }
 
        }
    }