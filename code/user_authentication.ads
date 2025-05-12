-- User_Authentication package specification
package User_Authentication is
   
   -- Maximum number of failed login attempts before temporary account lock
   Max_Failed_Attempts : constant Positive := 3;
   
   -- Minimum required password length
   Min_Password_Length : constant Positive := 8;
   
   -- User credentials data type
   type User_Credentials is private;
   
   -- Authentication exceptions
   Invalid_Credentials_Error : exception;
   Account_Locked_Error : exception;
   
   -- Create a new user credentials object
   function Create_User(Username : String; Password : String) return User_Credentials;
   
   -- Validate a user's credentials
   procedure Validate_Credentials(Username : in String; 
                                 Password : in String;
                                 Success : out Boolean);
   
   -- Authenticate a user (returns True if successful)
   function Authenticate(Username : String; 
                        Password : String) return Boolean;
   
   -- Record a failed login attempt
   procedure Record_Failed_Attempt(Username : String);
   
   -- Check if account is locked due to too many failed attempts
   function Is_Account_Locked(Username : String) return Boolean;
   
   -- Reset failed login attempts counter
   procedure Reset_Failed_Attempts(Username : String);
   
   -- Enforce password policy
   function Validate_Password_Policy(Password : String) return Boolean;
   
private
   
   -- Private implementation of User_Credentials type
   type User_Credentials is record
      Username : String(1..50);
      Password_Hash : String(1..64);
      Failed_Attempts : Natural := 0;
      Is_Locked : Boolean := False;
   end record;
   
end User_Authentication; 