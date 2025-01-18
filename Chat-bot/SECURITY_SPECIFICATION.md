# **Enhancing Security in FastAPI App**

## **Objective**
To address security issues in the FastAPI application related to file upload and access control.

---

## **Identified Issues**
1. **Unrestricted File Access**  
   Any user connected to the app can access uploaded files via the `/upload` endpoint.

2. **Lack of Group-Based Permissions**  
   The app does not support granting access to files based on specific user groups.

---

## **Proposed Solution**

### **User Authentication and Identification**
- Implement a user identification stage requiring a **userID**.
- Ensure only authenticated users can interact with the `/upload` endpoint or access files.

### **File Access Control**
- Allow file uploaders to specify access permissions:
  - **Group**: Accessible by users within a specified group.
  - **Private**: A group that includes only one userID (for user 17 it will be 10017).
  - **Public**: A group that includes all userIDs.

### **Admin Management**
- Introduce an **admin profile** to manage users and groups:
  - Add new users to groups or remove them as needed.
  - Maintain a secure record of groups and their members.
  - Admins have access to all files in the system and can modify file access control settings as needed.

---

## **Data Structure**
Maintain a table or dictionary containing user details:
- **`userID`**: The user’s unique identifier.
- **`groups`**: Comma-separated list of groups the user belongs to. All users will belong to to at least two groups: their private group and the public group.
- **`is_admin`**: Flag indicating whether the user has administrative privileges.

---

## **System Flow**

### **Authentication and Identification**

#### **Existing Users**
1. Prompt for a **userID**.
2. Check if the userID exists:
   - If the userID exists:
    - grant access to the app.
---

## **Admin Approval Workflow**
1. Admin receives a notification or a request to add the user to a group.
2. Admin can approve or reject the request.
3. Upon approval, the group is added to the user’s record, granting them access to group-specific files.

---

## **File Upload Workflow**

### **Authenticated Upload**
1. Prompt the uploader to select file access permissions:
   - **Private** (group 10000 + userID)
   - **Group-based** (specify group)
   - **Public** (group 10000)
2. Store metadata about file ownership and permissions.

### **File Access**
1. Verify user permissions before granting access.

---

### **Example Pseudo-Code for File Access**

```python
def can_access_file(user, file_metadata):
    # Check if the user is in the allowed group for the file
    group_id = file_metadata['group']  # Group ID is numeric
    
    if group_id == 10000:
        # Public group - anyone can access
        return True
    
    elif group_id == 10000 + user.id:
        # Private group - only the specific user can access
        return True
    
    elif group_id in user.groups:
        # Group-based - check if the user belongs to the group
        return True
    
    # Deny access if none of the conditions match
    return False
```

---

## **Admin Role**
Admin users can:
- Create, update, or delete groups.
- Approve user requests to join groups.
- Access all data files and share files or restrict file sharing.

---

## **System Workflow**

### **When a User Logs In**
1. Prompt for a userID:
   - If the userID exists:
     - Grant access to their files or relevant functionalities.
   - If the user doesn’t exist:
     - Prompt to create a new user.
     - Request admin approval for group assignment.

### **When Uploading a File**
1. Prompt the user to define access permissions:
   - **Private**, **Group**, or **Public**.

### **When Accessing Files**
1. Check user permissions against file metadata.
2. Deny access with a clear message if permissions are insufficient.

---

## **Next Steps**
1. Define API endpoints for user authentication, file upload, and file access.
2. Implement the admin interface for group and user management.
3. Integrate the new identification stage and ensure smooth workflow transitions.
4. Test the application to ensure all security measures are working as expected.
